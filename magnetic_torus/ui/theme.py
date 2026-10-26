"""Desktop visual theme."""

STYLE = """
QMainWindow, QWidget#workspace { background: #edf2f5; color: #203547; }
QWidget { font-family: 'Segoe UI'; font-size: 12px; color: #203547; }
QFrame#header { background: #ffffff; border-bottom: 1px solid #d5e0e7; }
QLabel#brand { font-size: 20px; font-weight: 650; color: #163c50; }
QLabel#subtitle { color: #647e8d; font-size: 11px; }
QLabel#badge { color: #317b80; background: #e3f1ee; padding: 5px 10px; border-radius: 8px; }
QLabel#message { padding: 6px 10px; background: #fff7e5; color: #806026; border-radius: 5px; }
QPushButton, QToolButton { padding: 6px 10px; border: 1px solid #d0dce4; border-radius: 5px; background: white; }
QPushButton:hover, QToolButton:hover { background: #e9f3f5; border-color: #82acb7; }
QPushButton:disabled, QToolButton:disabled { color: #a1afb7; background: #eef1f3; }
QPushButton#run { color: white; background: #187e86; border-color: #187e86; font-weight: 600; }
QPushButton#run:hover { background: #126670; }
QToolButton:checked { background: #d5eaeb; color: #14656d; border-color: #8bb9bb; }
QLineEdit, QComboBox { background: white; border: 1px solid #cedae3; padding: 5px 6px; border-radius: 4px; min-height: 17px; }
QLineEdit:focus, QComboBox:focus { border: 1px solid #258994; }
QScrollArea { border: none; background: #f7f9fb; }
QFrame#sidebar { background: #f7f9fb; border-right: 1px solid #d6e0e7; }
QFrame#section { border-bottom: 1px solid #dce5eb; }
QToolButton#sectionTitle { border: none; background: transparent; text-align: left; font-weight: 600; padding: 9px 0; }
QFrame#chart { background: white; border: 1px solid #d7e2e8; border-radius: 6px; }
QFrame#timeline { background: white; border: 1px solid #d7e2e8; border-radius: 5px; }
QProgressBar { border: none; background: #dfe8ec; border-radius: 3px; height: 6px; max-height: 6px; }
QProgressBar::chunk { background: #2a9c9c; border-radius: 3px; }
QSplitter::handle { background: #dce5ea; }
QPlainTextEdit { background: #142332; color: #d2e5ef; border: none; font-family: Consolas; font-size: 11px; }
QStatusBar { color: #627b8b; }
"""
