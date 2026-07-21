from pathlib import Path
from typing import Any
import mimetypes
import re
from urllib.parse import unquote

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from app.core.config import PHOTO_DIR
from app.services.indexer import index_folder


SCOPES = [
    "https://www.googleapis.com/auth/photospicker.mediaitems.readonly"
]

BASE_DIR = Path(__file__).resolve().parents[2]

CREDENTIALS_FILE = (
    BASE_DIR
    / "credentials"
    / "google_client_secret.json"
)

TOKEN_FILE = (
    BASE_DIR
    / "credentials"
    / "google_token.json"
)

PICKER_API_BASE_URL = "https://photospicker.googleapis.com/v1"


class GooglePhotosService:
    def get_credentials(self) -> Credentials:
        """
        Load existing Google credentials.

        If no valid token exists, open Google's local OAuth
        authorization flow and save the resulting token.
        """

        credentials: Credentials | None = None

        if TOKEN_FILE.exists():
            credentials = Credentials.from_authorized_user_file(
                str(TOKEN_FILE),
                SCOPES,
            )

        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
            self.save_credentials(credentials)

        if not credentials or not credentials.valid:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    f"Google OAuth credentials file was not found: "
                    f"{CREDENTIALS_FILE}"
                )

            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE),
                SCOPES,
            )

            credentials = flow.run_local_server(
                host="localhost",
                port=0,
                open_browser=True,
                authorization_prompt_message=(
                    "Open this URL in your browser to authorize "
                    "Google Photos:\n{url}"
                ),
                success_message=(
                    "Google Photos authorization completed. "
                    "You can close this browser tab."
                ),
            )

            self.save_credentials(credentials)

        return credentials

    def save_credentials(self, credentials: Credentials) -> None:
        """Save Google OAuth credentials locally."""

        TOKEN_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        TOKEN_FILE.write_text(
            credentials.to_json(),
            encoding="utf-8",
        )

    def authorization_headers(self) -> dict[str, str]:
        """Return authorization headers for Google Picker API calls."""

        credentials = self.get_credentials()

        return {
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json",
        }

    def connection_status(self) -> dict[str, Any]:
        """Return the current Google Photos connection status."""

        if not CREDENTIALS_FILE.exists():
            return {
                "configured": False,
                "authenticated": False,
                "message": (
                    "google_client_secret.json was not found"
                ),
            }

        if not TOKEN_FILE.exists():
            return {
                "configured": True,
                "authenticated": False,
                "message": (
                    "Google Photos is configured but not authenticated"
                ),
            }

        try:
            credentials = Credentials.from_authorized_user_file(
                str(TOKEN_FILE),
                SCOPES,
            )

            if credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
                self.save_credentials(credentials)

            return {
                "configured": True,
                "authenticated": bool(credentials.valid),
                "message": (
                    "Google Photos is connected"
                    if credentials.valid
                    else "Stored Google credentials are invalid"
                ),
            }

        except Exception as exc:
            return {
                "configured": True,
                "authenticated": False,
                "message": (
                    f"Could not read Google credentials: {exc}"
                ),
            }

    def authenticate(self) -> dict[str, Any]:
        """Run or verify Google authentication."""

        credentials = self.get_credentials()

        return {
            "authenticated": bool(credentials.valid),
            "message": "Google Photos authorization completed",
        }

    def create_picker_session(self) -> dict[str, Any]:
        """Create a new Google Photos Picker session."""

        response = requests.post(
            f"{PICKER_API_BASE_URL}/sessions",
            headers=self.authorization_headers(),
            json={},
            timeout=30,
        )

        self._raise_for_google_error(response)

        return response.json()

    def get_picker_session(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """Retrieve a Google Photos Picker session."""

        response = requests.get(
            f"{PICKER_API_BASE_URL}/sessions/{session_id}",
            headers=self.authorization_headers(),
            timeout=30,
        )

        self._raise_for_google_error(response)

        return response.json()

    def list_selected_media(
        self,
        session_id: str,
        page_size: int = 100,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        """List media selected by the user in a Picker session."""

        params: dict[str, Any] = {
            "sessionId": session_id,
            "pageSize": min(max(page_size, 1), 100),
        }

        if page_token:
            params["pageToken"] = page_token

        response = requests.get(
            f"{PICKER_API_BASE_URL}/mediaItems",
            headers=self.authorization_headers(),
            params=params,
            timeout=30,
        )

        self._raise_for_google_error(response)

        return response.json()

    def import_selected_media(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Download all selected image files and run the existing
        photo-indexing pipeline on the imported folder.
        """

        session = self.get_picker_session(session_id)

        if not session.get("mediaItemsSet", False):
            raise RuntimeError(
                "Photo selection is not complete. Open the Picker URL, "
                "select the photos and confirm your selection first."
            )

        import_directory = PHOTO_DIR / "google_photos"

        import_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        downloaded: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        page_token: str | None = None

        while True:
            selected_media = self.list_selected_media(
                session_id=session_id,
                page_size=100,
                page_token=page_token,
            )

            media_items = selected_media.get(
                "mediaItems",
                [],
            )

            for media_item in media_items:
                try:
                    result = self._download_media_item(
                        media_item=media_item,
                        destination_directory=import_directory,
                    )

                    if result["downloaded"]:
                        downloaded.append(result)
                    else:
                        skipped.append(result)

                except Exception as exc:
                    media_file = (
                        media_item.get("mediaFile")
                        or {}
                    )

                    errors.append(
                        {
                            "media_item_id": media_item.get("id"),
                            "filename": media_file.get("filename"),
                            "error": str(exc),
                        }
                    )

            page_token = selected_media.get(
                "nextPageToken"
            )

            if not page_token:
                break

        indexing_result = index_folder(import_directory)

        return {
            "session_id": session_id,
            "import_directory": str(import_directory),
            "downloaded_count": len(downloaded),
            "skipped_count": len(skipped),
            "error_count": len(errors),
            "downloaded": downloaded,
            "skipped": skipped,
            "errors": errors,
            "indexing_result": indexing_result,
        }

    def _download_media_item(
        self,
        media_item: dict[str, Any],
        destination_directory: Path,
    ) -> dict[str, Any]:
        """Download one selected Google Photos image."""

        media_file = media_item.get("mediaFile") or {}

        base_url = media_file.get("baseUrl")
        mime_type = media_file.get("mimeType", "")
        original_filename = media_file.get(
            "filename",
            "",
        )

        media_item_id = str(
            media_item.get("id") or "google-photo"
        )

        if not base_url:
            raise RuntimeError(
                "Selected media item does not contain a baseUrl"
            )

        if not mime_type.startswith("image/"):
            return {
                "downloaded": False,
                "media_item_id": media_item_id,
                "filename": original_filename,
                "reason": (
                    f"Unsupported media type: {mime_type}"
                ),
            }

        filename = self._safe_filename(
            original_filename=original_filename,
            media_item_id=media_item_id,
            mime_type=mime_type,
        )

        destination = self._unique_destination(
            directory=destination_directory,
            filename=filename,
        )

        # Request the original downloadable media bytes.
        download_url = f"{base_url}=d"

        response = requests.get(
            download_url,
            headers={
                "Authorization": (
                    f"Bearer {self.get_credentials().token}"
                )
            },
            timeout=120,
            stream=True,
        )

        self._raise_for_google_error(response)

        with destination.open("wb") as output_file:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    output_file.write(chunk)

        if destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)

            raise RuntimeError(
                "Google returned an empty media file"
            )

        return {
            "downloaded": True,
            "media_item_id": media_item_id,
            "filename": destination.name,
            "path": str(destination),
            "mime_type": mime_type,
            "size_bytes": destination.stat().st_size,
        }

    @staticmethod
    def _safe_filename(
        original_filename: str,
        media_item_id: str,
        mime_type: str,
    ) -> str:
        """Convert a Google filename into a safe local filename."""

        decoded_name = unquote(
            original_filename
        ).strip()

        if not decoded_name:
            extension = (
                mimetypes.guess_extension(mime_type)
                or ".jpg"
            )

            decoded_name = (
                f"google_{media_item_id}{extension}"
            )

        safe_name = re.sub(
            r"[^A-Za-z0-9._ -]+",
            "_",
            decoded_name,
        )

        safe_name = safe_name.strip(" .")

        if not safe_name:
            extension = (
                mimetypes.guess_extension(mime_type)
                or ".jpg"
            )

            safe_name = (
                f"google_{media_item_id}{extension}"
            )

        return safe_name

    @staticmethod
    def _unique_destination(
        directory: Path,
        filename: str,
    ) -> Path:
        """Prevent an imported photo from overwriting an existing file."""

        candidate = directory / filename

        if not candidate.exists():
            return candidate

        stem = candidate.stem
        suffix = candidate.suffix
        counter = 1

        while True:
            candidate = (
                directory
                / f"{stem}_{counter}{suffix}"
            )

            if not candidate.exists():
                return candidate

            counter += 1

    def delete_picker_session(
        self,
        session_id: str,
    ) -> None:
        """Delete a completed Google Photos Picker session."""

        response = requests.delete(
            f"{PICKER_API_BASE_URL}/sessions/{session_id}",
            headers=self.authorization_headers(),
            timeout=30,
        )

        self._raise_for_google_error(response)

    @staticmethod
    def _raise_for_google_error(
        response: requests.Response,
    ) -> None:
        """Raise a readable exception for unsuccessful Google responses."""

        if response.ok:
            return

        try:
            details = response.json()
        except ValueError:
            details = response.text

        raise RuntimeError(
            "Google Photos API error "
            f"{response.status_code}: {details}"
        )


google_photos_service = GooglePhotosService()