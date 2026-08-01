"""Small modal for correcting a recognition result (Phase 5, PR 4).

The user runs "Recognize", sees a prediction, and if it's wrong they open
this dialog. It shows what the model said plus its confidence, and lets
the user type the actual transcript. On accept, the tab persists the
correction via FeedbackStore.

Kept intentionally minimal: no stroke re-drawing, no image preview — the
canvas already shows what the user drew. The dialog's job is to capture
the corrected string; wiring to the store lives in the tab so the dialog
stays trivially testable and reusable if another surface (upload tab,
etc.) needs corrections later.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)


class CorrectionDialog(QDialog):
    def __init__(
        self,
        prediction: str,
        confidence: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Correct recognition")

        layout = QVBoxLayout(self)

        prediction_label = QLabel(
            f"Model predicted: <b>{prediction}</b> ({confidence:.0%} confidence)"
        )
        prediction_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(prediction_label)

        layout.addWidget(QLabel("Correct transcript:"))
        self._edit = QLineEdit(prediction)
        self._edit.selectAll()
        layout.addWidget(self._edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._edit.setFocus()

    def corrected_text(self) -> str:
        return self._edit.text().strip()
