from PySide6.QtWidgets import QDialog

from app.gui.widgets.correction_dialog import CorrectionDialog


def test_dialog_prefills_edit_with_prediction(qtbot):
    dialog = CorrectionDialog(prediction="helllo", confidence=0.42)
    qtbot.addWidget(dialog)

    assert dialog.corrected_text() == "helllo"


def test_dialog_returns_edited_text_stripped(qtbot):
    dialog = CorrectionDialog(prediction="foo", confidence=0.9)
    qtbot.addWidget(dialog)
    # Simulate the user replacing the text
    dialog._edit.setText("  hello  ")

    assert dialog.corrected_text() == "hello"


def test_accept_and_reject_use_qdialog_codes(qtbot):
    dialog = CorrectionDialog(prediction="x", confidence=0.5)
    qtbot.addWidget(dialog)
    # Sanity: our dialog uses QDialog's Accepted/Rejected codes, so the
    # tab can compare exec() against DialogCode.Accepted portably.
    assert CorrectionDialog.DialogCode.Accepted == QDialog.DialogCode.Accepted
    assert CorrectionDialog.DialogCode.Rejected == QDialog.DialogCode.Rejected
