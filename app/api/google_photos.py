from fastapi import APIRouter, HTTPException, Query

from app.services.google_photos_service import google_photos_service


router = APIRouter(
    prefix="/google-photos",
    tags=["Google Photos"],
)


@router.get("/status")
def google_photos_status() -> dict:
    return google_photos_service.connection_status()


@router.post("/authenticate")
def authenticate_google_photos() -> dict:
    try:
        return google_photos_service.authenticate()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@router.post("/sessions")
def create_picker_session() -> dict:
    try:
        session = google_photos_service.create_picker_session()

        return {
            "session_id": session.get("id"),
            "picker_uri": session.get("pickerUri"),
            "polling_config": session.get("pollingConfig"),
            "media_items_set": session.get(
                "mediaItemsSet",
                False,
            ),
            "raw_session": session,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@router.get("/sessions/{session_id}")
def get_picker_session(
    session_id: str,
) -> dict:
    try:
        return google_photos_service.get_picker_session(
            session_id
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@router.get("/sessions/{session_id}/media-items")
def list_selected_media(
    session_id: str,
    page_size: int = Query(
        default=100,
        ge=1,
        le=100,
    ),
    page_token: str | None = None,
) -> dict:
    try:
        return google_photos_service.list_selected_media(
            session_id=session_id,
            page_size=page_size,
            page_token=page_token,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@router.post("/sessions/{session_id}/import")
def import_selected_media(
    session_id: str,
) -> dict:
    try:
        return google_photos_service.import_selected_media(
            session_id
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@router.delete("/sessions/{session_id}")
def delete_picker_session(
    session_id: str,
) -> dict:
    try:
        google_photos_service.delete_picker_session(
            session_id
        )

        return {
            "deleted": True,
            "session_id": session_id,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc