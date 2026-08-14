import io
from types import SimpleNamespace

import whisper_flask2


class FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeModel:
    def transcribe(self, path: str, *, language: str, vad_filter: bool):
        assert path
        assert language == whisper_flask2.LANGUAGE
        assert vad_filter is True
        return (
            [FakeSegment("  hello "), FakeSegment("world  ")],
            SimpleNamespace(language="en", language_probability=0.98765),
        )


def client(monkeypatch):
    monkeypatch.setattr(whisper_flask2, "get_model", lambda: FakeModel())
    whisper_flask2.app.config["TESTING"] = True
    return whisper_flask2.app.test_client()


def test_missing_audio_is_rejected(monkeypatch):
    response = client(monkeypatch).post("/transcribe")
    assert response.status_code == 400


def test_unsupported_extension_is_rejected(monkeypatch):
    response = client(monkeypatch).post(
        "/transcribe",
        data={"audio": (io.BytesIO(b"data"), "payload.exe")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_transcription_contract(monkeypatch):
    response = client(monkeypatch).post(
        "/transcribe",
        data={"audio": (io.BytesIO(b"audio"), "sample.wav")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["text"] == "hello world"
    assert body["language"] == "en"
    assert body["language_probability"] == 0.9877
    assert body["elapsed_seconds"] >= 0
