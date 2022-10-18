"""SEGMENTERS MODULE"""


#! IMPORTS

import os
import sys
import inspect
import warnings

from types import FunctionType, MethodType
from typing import Iterable
from numpy.typing import NDArray

import numpy as np
import cv2

from PySide2.QtWidgets import (
    QMessageBox,
    QLabel,
    QShortcut,
    QWidget,
    QScrollArea,
    QHBoxLayout,
    QVBoxLayout,
    QDoubleSpinBox,
    QCheckBox,
    QColorDialog,
    QInputDialog,
    QSizePolicy,
    QPushButton,
    QTextEdit,
    QFileDialog,
    QProgressBar,
)
from PySide2.QtGui import (
    QFont,
    QImage,
    QPixmap,
    QKeySequence,
    QMouseEvent,
    QResizeEvent,
    QColor,
)
from PySide2.QtCore import Qt, QEvent, QSize, QPoint, QObject

from .assets import BACKWARD, DELETE, FORWARD, DOWNWARD, UPWARD, as_pixmap
from .segmenters import Segmenter
from .utils import Signal, check_type


#! GLOBAL VARIABLES


QFONT = QFont("Arial", 12)
QSIZE = 25

invalid = ["Segmenter", "NDArray", "Signal"]
segmenters_module = sys.modules["pylabeller.segmenters"]
SEGMENTERS = inspect.getmembers(segmenters_module, inspect.isclass)
SEGMENTERS = [i for i in SEGMENTERS if i[0] not in invalid]


#! GLOBAL FUNCTIONS


def ndarray2qpixmap(
    ndarray: NDArray,
    frmt: QImage.Format = QImage.Format_RGBA8888,
) -> QPixmap:
    """
    return the pixmap corresponding to the NDArray provided.

    Parameters
    ----------
    ndarray: NDArray
        a 3D array with shape and dtype aligned with the given format.
        By default, ndarray is espected to have dtype=uint8 and RGBA color
        distribution.

    frmt: QtGui.QImage.Format
        the image format. By default an RGB format is used.

    Returns
    -------
    qpix: QtGui.QPixmap
        the pixmap corresponding to the array.
    """
    # check the entries
    check_type(ndarray, np.ndarray)
    assert ndarray.ndim >= 3, "ndarray must be a 3D+ NDArray."
    check_type(frmt, QImage.Format)

    # transform ndarray to pixmap
    shape = ndarray.shape
    qimg = QImage(
        ndarray,
        shape[1],
        shape[0],
        shape[-1] * shape[1],
        frmt,
    )
    return QPixmap(qimg)


def make_shortcut(
    shortcut: str | Qt.Key,
    parent: QWidget,
    action: FunctionType | MethodType,
) -> QShortcut:
    """
    create a new shortcut.

    Parameters
    ----------
    shortcut: str | Qt.Key
        the shortcut to be linked.

    parent: QWidget
        the widget being the parent of the shortcut.

    action: FunctionType | MethodType
        the action triggered by the shortcut.

    Returns
    -------
    shortcut: QShortcut
        the created shortcut.
    """
    check_type(shortcut, (str, Qt.Key))
    check_type(parent, QWidget)
    check_type(action, (FunctionType, MethodType))
    out = QShortcut(QKeySequence(shortcut), parent)
    out.activated.connect(action)
    return out


def get_label(text: str) -> QLabel:
    """
    return a QLabel formatted with the given text.

    Parameters
    ----------
    text: str
        The label text.

    Returns
    -------
    label: QtWidgets.QLabel
        the label object.
    """
    out = QLabel(text)
    out.setFont(QFONT)
    out.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
    return out


#! CLASSES


class FlatButton(QLabel):
    """
    create a custom flat button with the ability to
    discriminate between clicking events.

    Parameters
    ----------
    text: str
        the text to be shown by the button.

    icon: str | None
        the name of the icon to be used on this button. The icon is retrieved
        from the "assets" folder of the module.
        If None, no icon is used.

    tooltip: str | None
        the tooltip to be shown by the button. If None, no tooltip is provided.

    click_action: FunctionType | MethodType | None
        the function to be linked to the click action of the button.

    click_shortcut: str
        the action shortcut
    """

    # ****** SIGNALS ****** #

    _clicked = None

    # ****** VARIABLES ****** #

    _click_function = None
    _click_shortcut = None

    # ****** CONSTRUCTOR ****** #

    def __init__(
        self,
        text: str | None = None,
        icon: str | None = None,
        tooltip: str | None = "",
        fun: FunctionType | MethodType | None = None,
        shortcut: str | None = None,
    ):
        super().__init__(text)
        self.setFont(QFONT)
        self.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self._clicked = Signal()

        # set the input data
        if icon is not None:
            if not os.path.exists(icon):
                warnings.warn(f"{icon} does not exists.")
        if tooltip is not None:
            self.setToolTip(tooltip)
        self.setFont(QFONT)

        # connect to the signals
        if fun is not None:
            self.set_click_fun(fun)
        if shortcut is not None:
            self.set_shortcut(shortcut)

        # set icon and tooltip
        self.set_icon(icon)
        self.setToolTip(tooltip)

    # ****** SETTERS ****** #

    def set_icon(self, icon: str | NDArray | None) -> None:
        """set the label icon as pixmap."""
        if icon is None:
            self.setPixmap(None)
        else:
            check_type(icon, (str, np.ndarray))
            if isinstance(icon, str):
                qpix = as_pixmap(icon)
            else:
                qpix = ndarray2qpixmap(icon)
            self.setPixmap(qpix.scaled(QSIZE, QSIZE))

    def set_click_fun(self, fun: FunctionType | MethodType) -> None:
        """
        setup the left-click action.

        Parameters
        ----------
        fun: FunctionType | MethodType
            the action to be linked to left-click mouse event.
        """
        check_type(fun, (FunctionType, MethodType))
        self._click_function = fun
        self._clicked.connect(self._click_function)

    def set_shortcut(self, shortcut: str) -> None:
        """set the shortcut linked to the button."""
        self._click_shortcut = make_shortcut(
            shortcut=shortcut,
            parent=self,
            action=self._click_function,
        )

    # ****** PROPERTIES ****** #

    @property
    def icon(self) -> QPixmap | None:
        """return the icon installed on the button."""
        return self.pixmap()

    @property
    def click_function(self) -> FunctionType | MethodType | None:
        """return the left click action."""
        return self._click_function

    @property
    def clicked(self) -> Signal | None:
        """return the left click signal."""
        return self._clicked

    @property
    def shortcut(self) -> QShortcut | None:
        """return the shortcut linked to the object."""
        return self._click_shortcut

    @property
    def drop_action(self):
        """drop action"""
        return self._drop_action

    # ****** EVENT HANDLES ****** #

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """handle mouse double clicking event"""
        if event.button() == Qt.LeftButton:
            self._clicked.emit(self)


class TextSpinBox(QWidget):
    """
    create a custom widget where a spinbox is associated to a label and to a
    checkbox.

    Parameters
    ----------
    start_value: int | float
        the starting value of the slider

    min_value: int | float
        the minimum slider value

    max_value: int | float
        the maximum slider value

    step: int | float
        the step size increment/decrement

    decimals: int
        the number of decimals to be used for rendering the data.

    descriptor: str
        the descriptor of the values.
    """

    # ****** SIGNALS ****** #

    _value_changed = None
    _state_changed = None

    # ****** VARIABLES ****** #

    _label = None
    _spinbox = None

    # ****** CONSTRUCTOR ****** #

    def __init__(
        self,
        value: int | float,
        min_value: int | float,
        max_value: int | float,
        step: int | float,
        decimals: int,
        descriptor: str,
    ) -> None:
        super().__init__()

        # setup the widget
        self._spinbox = QDoubleSpinBox()
        self._spinbox.setFont(QFONT)
        self._label = QLabel()
        self._label.setFont(QFONT)
        self._checkbox = QCheckBox()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._label)
        layout.addWidget(self._spinbox)
        self.setLayout(layout)

        # setup the initial values
        self.set_min_value(min_value)
        self.set_max_value(max_value)
        self.set_step(step)
        self.set_decimals(decimals)
        self.set_value(value)
        self.set_descriptor(descriptor)

        # setup the connections
        self._value_changed = Signal()
        self._state_changed = Signal()
        self._spinbox.valueChanged.connect(self._on_value_changed)

    # ****** SETTERS ****** #

    def set_value(self, value: int | float) -> None:
        """set the actual value."""
        check_type(value, (int, float))
        val = float(value)
        self._spinbox.setValue(val)

    def set_min_value(self, value: int | float) -> None:
        """set the min value."""
        check_type(value, (int, float))
        val = float(value)
        self._spinbox.setMinimum(val)

    def set_max_value(self, value: int | float) -> None:
        """set the max value."""
        check_type(value, (int, float))
        val = float(value)
        self._spinbox.setMaximum(val)

    def set_step(self, value: int | float) -> None:
        """set the step value."""
        check_type(value, (int, float))
        val = float(value)
        self._spinbox.setSingleStep(val)

    def set_decimals(self, value: int) -> None:
        """set the step value."""
        check_type(value, int)
        self._spinbox.setDecimals(value)

    def set_descriptor(self, value: str) -> None:
        """set the descriptor."""
        check_type(value, str)
        self._label.setText(value)

    # ****** PROPERTIES ****** #

    @property
    def value(self) -> float:
        """return the object value"""
        return self._spinbox.value()

    @property
    def min_value(self) -> float:
        """return the object min_value"""
        return self._spinbox.minimum()

    @property
    def max_value(self) -> float:
        """return the object max_value"""
        return self._spinbox.maximum()

    @property
    def decimals(self) -> int:
        """return the object decimals"""
        return self._spinbox.decimals()

    @property
    def step(self) -> float:
        """return the object step"""
        return self._spinbox.singleStep()

    @property
    def descriptor(self) -> float:
        """return the object descriptor"""
        return self._label.text()

    @property
    def value_changed(self) -> Signal:
        """return the value_changed signal"""
        return self._value_changed

    # ****** EVENT HANDLERS ****** #

    def _on_value_changed(self, value: float) -> None:
        """handle the update of the spinbox value."""
        self.value_changed.emit(value)


class SegmenterWidget(QWidget):
    """
    Widget allowing the customization of the Segmenter object.

    Parameters
    ----------
    segmenter: Segmenter
        the segmenter linked to the widget.
    """

    # ****** SIGNALS ****** #

    _color_changed = None
    _name_changed = None
    _linewidth_changed = None
    _linewidth_checked = None
    _fontsize_changed = None
    _fontsize_checked = None
    _checked = None
    _deleted = None
    _move_downward = None
    _move_upward = None
    _text_checked = None

    # ****** VARIABLES ****** #

    _name_button = None
    _color_button = None
    _linewidth_box = None
    _fontsize_box = None
    _delete_button = None
    _check_box = None
    _segmenter = None
    _options_widget = None
    _upward_button = None
    _downward_button = None
    _text_checkbox = None

    # ****** CONSTRUCTOR ****** #

    def __init__(self, segmenter: Segmenter) -> None:
        super().__init__()

        # signals
        self._color_changed = Signal()
        self._name_changed = Signal()
        self._linewidth_changed = Signal()
        self._linewidth_checked = Signal()
        self._fontsize_changed = Signal()
        self._fontsize_checked = Signal()
        self._checked = Signal()
        self._deleted = Signal()
        self._move_upward = Signal()
        self._move_downward = Signal()
        self._text_checked = Signal()

        # name
        self._name_button = FlatButton(
            text="SEGMENTER",
            icon=None,
            tooltip="Rename",
            fun=self._name_clicked,
        )
        self._name_button.setFixedHeight(QSIZE)
        align_right = Qt.AlignRight | Qt.AlignVCenter
        self._name_button.setAlignment(align_right)

        # color
        self._color_button = FlatButton(
            text=None,
            icon=None,
            tooltip="Change the color",
            fun=self._color_clicked,
        )
        self._color_button.setFixedSize(QSIZE, QSIZE)

        # check button
        self._check_box = QCheckBox()
        self._check_box.setCheckable(True)
        self._check_box.setChecked(False)
        self._check_box.setFixedSize(QSIZE, QSIZE)
        self._check_box.setToolTip("Activate/Deactivate the segmenter")
        self._check_box.stateChanged.connect(self._check_clicked)

        # movement box
        self._upward_button = FlatButton(
            text=None,
            icon=UPWARD,
            tooltip="Move up",
            fun=self._on_move_upward,
        )
        self._downward_button = FlatButton(
            text=None,
            icon=DOWNWARD,
            tooltip="Move down",
            fun=self._on_move_downward,
        )
        move_layout = QHBoxLayout()
        move_layout.setContentsMargins(0, 0, 0, 0)
        move_layout.setSpacing(0)
        move_layout.addWidget(self._upward_button)
        move_layout.addWidget(self._downward_button)
        move_button = QWidget()
        move_button.setLayout(move_layout)
        move_button.setFixedSize(2 * QSIZE, QSIZE)

        # main line
        upper_layout = QHBoxLayout()
        upper_layout.addWidget(move_button)
        upper_layout.addWidget(self._name_button)
        upper_layout.addWidget(self._color_button)
        upper_layout.addWidget(self._check_box)
        upper_layout.setContentsMargins(0, 0, 0, 0)
        upper_layout.setSpacing(5)
        upper_widget = QWidget()
        upper_widget.setLayout(upper_layout)

        # delete
        self._delete_button = FlatButton(
            text=None,
            icon=DELETE,
            tooltip="Remove the Segmenter",
            fun=self._delete_clicked,
        )
        self._delete_button.setFixedSize(QSIZE, QSIZE)

        # linewidth
        self._linewidth_box = TextSpinBox(1, 1, 20, 1, 0, "Linewidth")
        self._linewidth_box.value_changed.connect(self._on_linewidth_changed)
        self._linewidth_box.setFixedHeight(QSIZE)

        # fontsize
        self._fontsize_box = TextSpinBox(2, 1, 20, 1, 0, "Fontsize")
        self._fontsize_box.value_changed.connect(self._on_fontsize_changed)
        self._fontsize_box.setFixedHeight(QSIZE)

        # text checkbox
        self._text_checkbox = QCheckBox()
        self._text_checkbox.setChecked(True)
        self._text_checkbox.setFont(QFONT)
        self._text_checkbox.setToolTip("Enable/disable text view.")
        self._text_checkbox.stateChanged.connect(self._on_text_checked)

        # options layout
        lower_layout = QHBoxLayout()
        lower_layout.addWidget(self._delete_button)
        lower_layout.addWidget(self._text_checkbox)
        lower_layout.addWidget(self._fontsize_box)
        lower_layout.addWidget(self._linewidth_box)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        lower_layout.setSpacing(5)
        self._options_widget = QWidget()
        self._options_widget.setLayout(lower_layout)
        self._options_widget.setVisible(False)

        # central layout
        central_layout = QVBoxLayout()
        central_layout.addWidget(upper_widget)
        central_layout.addWidget(self._options_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(2)
        self.setLayout(central_layout)

        # set the segmenter
        self.set_segmenter(segmenter)

        # set the minimum size
        self.setBaseSize(self.minimumSizeHint())
        self.sizePolicy().setHorizontalPolicy(QSizePolicy.Minimum)

    # ****** SETTERS ****** #

    def set_segmenter(self, segmenter: Segmenter) -> None:
        """
        set a new segmenter for the widget.

        Parameters
        ----------
        segmenter: Segmenter
            a new segmenter.
        """
        check_type(segmenter, Segmenter)
        self._segmenter = segmenter

        # update the widget options
        self.set_color(segmenter.color)
        self.set_name(segmenter.name)
        self.set_linewidth(segmenter.linewidth)
        self.set_fontsize(segmenter.fontsize)

    def set_color(self, rgba: Iterable | None = None) -> None:
        """
        set the required rgba color.

        Parameters
        ----------
        rgb: Iterable | None
            the 4 elements tuple defining a color.
        """

        # check the input
        if rgba is not None:
            msg = "color must be an len=4 iterable of int(s) in the [0-255]"
            msg += " range."
            if not isinstance(rgba, Iterable) or not len(rgba) == 4:
                raise TypeError(msg)
            if not all(0 <= i <= 255 for i in rgba):
                raise ValueError(msg)
            if not all(isinstance(i, int) for i in rgba):
                raise ValueError(msg)
        else:
            rgba = tuple(np.random.randint(0, 256) for _ in range(4))

        # generate an ndarray with the given color
        self._segmenter.set_color(rgba)
        self._color_button.setFixedSize(QSIZE, QSIZE)
        self._color_button.set_icon(self.segmenter.icon((QSIZE, QSIZE)))
        self.color_changed.emit(self, rgba)

    def set_name(self, name: str) -> None:
        """rename the current label."""
        old = self._name_button.text()
        self._name_button.setText(name)
        self._segmenter.set_name(name)
        self.name_changed.emit(self, old, name)

    def set_linewidth(self, linewidth: int) -> None:
        """set the segmenterwidget linewidth reference value."""
        check_type(linewidth, int)
        self._linewidth_box.set_value(linewidth)
        self._segmenter.set_linewidth(linewidth)
        self.linewidth_changed.emit(self, self._linewidth_box.value)

    def set_fontsize(self, fontsize: int) -> None:
        """set the segmenterwidget fontsize reference value."""
        check_type(fontsize, int)
        self._fontsize_box.set_value(fontsize)
        self._segmenter.set_fontsize(fontsize)
        self.fontsize_changed.emit(self, self._fontsize_box.value)

    def set_move_upward_enabled(self, enabled: bool) -> None:
        """enable/disable the upward move button"""
        check_type(enabled, bool)
        self._upward_button.setEnabled(enabled)

    def set_move_downward_enabled(self, enabled: bool) -> None:
        """enable/disable the downward move button"""
        check_type(enabled, bool)
        self._downward_button.setEnabled(enabled)

    # ****** GETTERS ****** #

    @property
    def name_changed(self) -> Signal | None:
        """return the name change signal"""
        return self._name_changed

    @property
    def color_changed(self) -> Signal | None:
        """return the color change signal"""
        return self._color_changed

    @property
    def linewidth_changed(self) -> Signal | None:
        """return the linewidth change signal"""
        return self._linewidth_changed

    @property
    def linewidth_checked(self) -> Signal | None:
        """return the linewidth check signal"""
        return self._linewidth_checked

    @property
    def fontsize_changed(self) -> Signal | None:
        """return the fontsize change signal"""
        return self._fontsize_changed

    @property
    def fontsize_checked(self) -> Signal | None:
        """return the fontsize check signal"""
        return self._fontsize_checked

    @property
    def checked(self) -> Signal | None:
        """return the checked signal"""
        return self._checked

    @property
    def deleted(self) -> Signal | None:
        """return the delete signal"""
        return self._deleted

    @property
    def segmenter(self) -> Segmenter | None:
        """return the segmenter"""
        return self._segmenter

    @property
    def fontsize_is_checked(self) -> bool:
        """return the state of the fontsize textspinbox"""
        return self._fontsize_box.is_checked()

    @property
    def linewidth_is_checked(self) -> bool:
        """return the state of the linewidth textspinbox"""
        return self._linewidth_box.is_checked()

    @property
    def move_downward(self) -> Signal:
        """return the state_changed signal"""
        return self._move_downward

    @property
    def move_upward(self) -> Signal:
        """return the state_changed signal"""
        return self._move_upward

    @property
    def text_checked(self) -> Signal:
        """return the text_checked signal"""
        return self._text_checked

    # ****** EVENT HANDLERS ****** #

    def _color_clicked(self, source: FlatButton) -> None:
        """select and set the desired color."""
        self.change_color()

    def _name_clicked(self, source: FlatButton) -> None:
        """handle the single clicking action."""
        self.change_name()

    def _check_clicked(self, source: FlatButton):
        """handle the double clicking action on the segmenter widget"""
        self.checked.emit(self)

    def _delete_clicked(self, source: FlatButton) -> None:
        """handle the delete button clicking."""
        self.deleted.emit(self)

    def _on_linewidth_changed(self, value: int | float) -> None:
        """handle the linewidth change."""
        self.segmenter.set_linewidth(int(value))
        self.linewidth_changed.emit(self, int(value))

    def _on_linewidth_checked(self, value: bool) -> None:
        """handle the state of the linewidth SpinCheck"""
        self.linewidth_checked.emit(self, value)

    def _on_fontsize_changed(self, value: int | float) -> None:
        """handle the fontsize change."""
        self.segmenter.set_fontsize(int(value))
        self.fontsize_changed.emit(self, int(value))

    def _on_fontsize_checked(self, value: bool) -> None:
        """handle the state of the fontsize SpinCheck"""
        self.fontsize_checked.emit(self, value)

    def _on_move_upward(self, source: FlatButton) -> None:
        """handle the upward button action."""
        self.move_upward.emit(self)

    def _on_move_downward(self, source: FlatButton) -> None:
        """handle the downward button action."""
        self.move_downward.emit(self)

    def _on_text_checked(self, value: bool) -> None:
        """enable/disable the fontsize TextSpinbox"""
        self._fontsize_box.setEnabled(self._text_checkbox.isChecked())
        self.text_checked.emit(self)

    def enterEvent(self, event: QEvent) -> None:
        """make the delete button visible."""
        self._options_widget.setVisible(True)
        return super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        """hide the delete button."""
        self._options_widget.setVisible(False)
        return super().leaveEvent(event)

    # ****** METHODS ****** #

    def minimumSizeHint(self) -> QSize:
        """return the minimum size hint for this widget."""
        width = self._options_widget.minimumSizeHint().width()
        height = super().minimumSizeHint().height()
        return QSize(width, height)

    def is_checked(self) -> bool:
        """return whether the Segmenter is active or not."""
        return self._check_box.isChecked()

    def is_text_enabled(self) -> bool:
        """return whether the text_checkbox is checked or not."""
        return self._text_checkbox.isChecked()

    def change_color(self) -> None:
        """method allowing the change of the Segmenter color."""
        color = QColorDialog.getColor(initial=QColor(*self.segmenter.color))
        if color.isValid():
            self.set_color(color.toTuple())

    def change_name(self) -> None:
        """method allowing the change of the Segmenter name."""
        text, done = QInputDialog.getText(
            self,
            "rename",
            "Write here the new label name",
        )
        if done:
            self.set_name(text)

    def click(self) -> None:
        """force the check/uncheck of the segmenter"""
        state = False if self._check_box.isChecked() else True
        self._check_box.setChecked(state)
        self.checked.emit(self)


class SegmenterPaneWidget(QWidget):
    """
    SegmenterWidget(s) grouper.

    Parameters
    ----------
    segmenters: Iterable[Segmenter] | Segmenter | None
        a list of segmenters to be rendered.
    """

    # ****** SIGNALS ****** #

    _added = None
    _removed = None
    _text_checked = None

    # ****** VARIABLES ****** #

    _segmenters = []
    _add_button = None
    _container = None
    _shortcuts = []
    _scroll = None

    # ****** CONSTRUCTOR ****** #

    def __init__(
        self,
        sgmntrs: Iterable[Segmenter] | Segmenter | None = None,
    ) -> None:
        super().__init__()
        self._added = Signal()
        self._removed = Signal()
        self._text_checked = Signal()
        self._shortcuts = []

        # setup the add button
        self._add_button = QPushButton("Add New")
        self._add_button.setToolTip("Add a new Segmenter.")
        self._add_button.clicked.connect(self._add_clicked)
        self._add_button.setFont(QFONT)
        self._add_button.setFixedHeight(QSIZE)
        min_width = self._add_button.minimumSizeHint().width()
        self._add_button.setFixedWidth(min_width)

        # set the top widget
        descriptor = QLabel("SEGMENTER LIST")
        descriptor.setFont(QFONT)
        descriptor.setFixedHeight(QSIZE)
        descriptor.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        top_layout = QHBoxLayout()
        top_layout.addWidget(descriptor)
        top_layout.addWidget(self._add_button)
        top_layout.setSpacing(5)
        top_widget = QWidget()
        top_widget.setLayout(top_layout)
        top_widget.setFixedHeight(top_widget.minimumSizeHint().height())

        # set the segmenters widget
        sgm_layout = QVBoxLayout()
        sgm_layout.setContentsMargins(0, 0, 0, 0)
        sgm_layout.setSpacing(5)
        self._container = QWidget()
        self._container.setLayout(sgm_layout)

        # setup a scroll area for the container
        column = QWidget()
        column.setFixedWidth(20)
        inner_layout = QHBoxLayout()
        inner_layout.setSpacing(0)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.addWidget(self._container)
        inner_layout.addWidget(column)
        inner_widget = QWidget()
        inner_widget.setLayout(inner_layout)
        self._scroll = QScrollArea()
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(inner_widget)
        self._scroll.verticalScrollBar().setFixedWidth(20)

        # add the segmenters
        if sgmntrs is not None:
            sgm = sgmntrs if isinstance(sgmntrs, Iterable) else [sgmntrs]
            for segmenter in sgm:
                check_type(segmenter, Segmenter)
                self.add_segmenter(segmenter)
        self._update_shortcuts()
        self._update_move_buttons()

        # setup the widget layout
        layout = QVBoxLayout()
        layout.addWidget(top_widget)
        layout.addWidget(self._scroll)
        self.setLayout(layout)

    # ****** GETTERS ****** #

    @property
    def shortcuts(self) -> list[QShortcut]:
        """return the list of shortcuts."""
        return self._shortcuts

    @property
    def added(self) -> Signal:
        """return the added signal"""
        return self._added

    @property
    def removed(self) -> Signal:
        """return the removed signal"""
        return self._removed

    @property
    def segmenters(self) -> list[SegmenterWidget]:
        """return the list of available segmenter widgets."""
        return self._segmenters

    @property
    def active_id(self) -> int | None:
        """return the index of the actual active SegmenterWidget or None."""
        active = np.where([i.is_checked() for i in self.segmenters])[0]
        return active[0] if len(active) > 0 else None

    @property
    def active_widget(self) -> SegmenterWidget | None:
        """return the actual active SegmenterWidget or None."""
        idx = self.active_id
        return self.segmenters[idx] if idx is not None else None

    @property
    def active_segmenter(self) -> Segmenter | None:
        """return the actual active Segmenter or None."""
        segmenter = self.active_widget
        if segmenter is not None:
            segmenter = segmenter.segmenter
        return segmenter

    @property
    def text_checked(self) -> Signal | None:
        """return the text_checked signal"""
        return self._text_checked

    # ****** EVENT HANDLERS ****** #

    def _update_state(self, source: SegmenterWidget) -> None:
        """handle the clicking of any SegmenterWidget within the group."""
        if source.is_checked():
            for i in self.segmenters:
                if i != source and i.is_checked():
                    i.click()
                    # i._check_box.setChecked(False)

    def _delete_segmenter(self, source: FlatButton) -> None:
        """handle the removal of a SegmenterWidget from the group."""

        # remove the segmenter from the list
        index = [i for i, v in enumerate(self.segmenters) if v == source]
        if len(index) > 0:
            self.delete_segmenter(index[0])

        # update the layout
        self._update_layout()

    def _add_clicked(self, source: FlatButton) -> None:
        """handle the removal of a SegmenterWidget from the group."""
        self.new_segmenter()

    def _shift_up(self, source: SegmenterWidget) -> None:
        """handle the shift up of the segmenter"""
        index = None
        for i, v in enumerate(self.segmenters):
            if v.segmenter.name == source.segmenter.name:
                index = i
                break
        if index is not None and index > 0:
            pre = self.segmenters[: (index - 1)]
            sgm = [self.segmenters[index]]
            pst = [self.segmenters[index - 1]] + self.segmenters[(index + 1) :]
            self._segmenters = pre + sgm + pst
            self._update_move_buttons()
        self._update_layout()

    def _shift_down(self, source: SegmenterWidget) -> None:
        """handle the shift down of the segmenter"""
        index = None
        for i, v in enumerate(self.segmenters):
            if v.segmenter.name == source.segmenter.name:
                index = i
                break
        if index is not None and index < len(self.segmenters) - 1:
            pre = self.segmenters[:index] + [self.segmenters[index + 1]]
            sgm = [self.segmenters[index]]
            pst = self.segmenters[index + 2 :]
            self._segmenters = pre + sgm + pst
            self._update_move_buttons()
        self._update_layout()

    def _on_text_checked(self, source: SegmenterWidget) -> None:
        """handle the check/uncheck of the text_checkbox of one widget"""
        self.text_checked.emit(source)

    # ****** METHODS ****** #

    def _update_layout(self) -> None:
        """update the layout according to the actual segmenters listed."""

        # update the layout
        layout = self._container.layout()
        while layout.count() > 0:
            item = layout.itemAt(0)
            layout.removeItem(item)
        for widget in self.segmenters:
            layout.addWidget(widget)
        layout.addStretch()
        self._update_shortcuts()
        self._update_move_buttons()

        if len(self.segmenters) > 0:
            width = max([i.minimumSizeHint().width() for i in self.segmenters])
        else:
            width = self._scroll.minimumSizeHint().width()
        self._scroll.setMinimumWidth(width + 20)

    def new_segmenter(self) -> None:
        """add a novel segmenter"""
        # let the user select the new segmenter type
        items = [i[0] for i in SEGMENTERS]
        item, done = QInputDialog.getItem(
            self,
            "Segmenter type",
            "Set the new segmenter type",
            items,
            0,
            False,
        )

        # add the segmenter
        if done and isinstance(item, str):
            idx = np.where([i == item for i in items])[0][0]
            cls = SEGMENTERS[idx][1]
            self.add_segmenter(
                segmenter=cls(name=f"SEGMENTER{len(self.segmenters) + 1}"),
                index=0,
            )
            self.segmenters[0].click()

    def add_segmenter(
        self,
        segmenter: Segmenter,
        index: int | None = None,
    ) -> None:
        """
        add a SegmenterWidget to the group.

        Parameters
        ----------
        segmenter: Segmenter
            a Segmenter object to be appended to the group.

        index: int | None
            the position of the list at which include the Segmenter.
        """

        # check the entries
        check_type(segmenter, Segmenter)
        isin = any(segmenter.name == i.segmenter.name for i in self.segmenters)
        if not isin:

            # convert the segmenter to SegmenterWidget and
            # connect with _update_state and _delete_segmenter actions
            widget = SegmenterWidget(segmenter)
            widget.checked.connect(self._update_state)
            widget.deleted.connect(self._delete_segmenter)
            widget.move_upward.connect(self._shift_up)
            widget.move_downward.connect(self._shift_down)
            widget.text_checked.connect(self._on_text_checked)

            # append the segmenter
            if index is None:
                self._segmenters += [widget]
            elif isinstance(index, int):
                self._segmenters.insert(index, widget)
            else:
                raise TypeError(f"{index} must be an {int} instance.")

            # update
            self._update_layout()
            self.added.emit(widget)

    def delete_segmenter(self, index: int) -> None:
        """
        delete a SegmenterWidget from the group.

        Parameters
        ----------
        index: int
            the position of the widget to be removed from the list.
        """
        if isinstance(index, int):
            wdg = self.segmenters.pop(index)
            wdg.setVisible(False)
            self._update_layout()
            self.removed.emit(wdg)
        else:
            raise TypeError(f"{index} must be an {int} instance.")

    def _update_shortcuts(self) -> None:
        """update the shortcuts linked to the object."""
        for shortcut in self._shortcuts:
            shortcut.deleteLater()
        self._shortcuts = []
        for i, segmenter in enumerate(self.segmenters):
            if i == 9:
                msg = "No more than 9 segmenters can be linked to numerical "
                msg += "shortcuts."
                QMessageBox.warning(self, "WARNING", msg)
            elif i < 9:
                key = Qt.Key(Qt.Key_1 + i)
                short = make_shortcut(key, self, segmenter.click)
                self._shortcuts += [short]
        add_short = make_shortcut(Qt.Key_Plus, self, self.new_segmenter)
        self._shortcuts += [add_short]

    def _update_move_buttons(self) -> None:
        """
        update the move button are enabled/disabled according to the position
        of the segmenter in the list.
        """
        n = len(self.segmenters)
        for i, segmenter in enumerate(self.segmenters):
            segmenter.set_move_downward_enabled(i < n - 1)
            segmenter.set_move_upward_enabled(i > 0)


class CounterWidget(QWidget):
    """
    Image counter widget.

    Parameters
    ----------
    max_counter: int
        the maximum number of elements to be counted.

    start_value: int | None
        the starting counter value. if None, it is initialized to zero.
    """

    # ****** SIGNALS ****** #

    _move_forward = None
    _move_backward = None

    # ****** VARIABLES ****** #

    _label = None
    _forward_button = None
    _backward_button = None
    _counter_label = None
    _max_counter_label = None

    # ****** CONSTRUCTOR ****** #

    def __init__(
        self,
        text: str,
        max_counter: int,
        start_value: int | None = None,
    ) -> None:
        super().__init__()
        self._move_forward = Signal()
        self._move_backward = Signal()

        # setup the max counter
        check_type(max_counter, int)
        self._max_counter_label = get_label(str(max_counter))

        # setup the counter
        self._counter_label = get_label("")
        if start_value is None:
            self.set_counter(1)
        else:
            check_type(start_value, int)
            self.set_counter(start_value)

        # setup the text
        check_type(text, str)
        self._label = get_label(text)

        # setup the forward button
        size = self._counter_label.minimumSizeHint().height()
        self._forward_button = QPushButton()
        self._forward_button.setShortcut(QKeySequence(Qt.Key_Right))
        self._forward_button.clicked.connect(self._forward_pressed)
        self._forward_button.setIcon(as_pixmap(FORWARD).scaled(size, size))
        self._forward_button.setFixedSize(size, size)
        self._forward_button.setToolTip("Next Frame")
        self._forward_button.setFont(QFONT)

        # setup the backward button
        self._backward_button = QPushButton()
        self._backward_button.setShortcut(QKeySequence(Qt.Key_Left))
        self._backward_button.clicked.connect(self._backward_pressed)
        self._backward_button.setIcon(as_pixmap(BACKWARD).scaled(size, size))
        self._backward_button.setFixedSize(size, size)
        self._backward_button.setToolTip("Previous Frame")
        self._backward_button.setFont(QFONT)
        self._backward_button.setEnabled(False)

        # setup the whole widget layout
        layout = QHBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        layout.addWidget(self._label)
        layout.addStretch()
        layout.addWidget(self._backward_button)
        layout.addWidget(self._counter_label)
        layout.addWidget(get_label(" / "))
        layout.addWidget(self._max_counter_label)
        layout.addWidget(self._forward_button)
        layout.addStretch()
        self.setLayout(layout)

    # ****** SETTERS ****** #

    def set_counter(self, cnt: int) -> None:
        """set the counter to the required value."""
        check_type(cnt, int)
        digits = len(self._max_counter_label.text())
        frm = "{:0" + str(digits) + "d}"
        self._counter_label.setText(frm.format(cnt))

    def set_max_counter(self, cnt: int) -> None:
        """set the max counter to the required value."""
        check_type(cnt, int)
        self._max_counter_label.setText(str(cnt))

    # ****** GETTERS ****** #

    @property
    def move_forward(self) -> Signal | None:
        """return the move forward signal"""
        return self._move_forward

    @property
    def move_backward(self) -> Signal | None:
        """return the move backward signal"""
        return self._move_backward

    @property
    def counter(self) -> int:
        """return the actual counter value."""
        return int(self.counter_label.text())

    @property
    def max_counter(self) -> int:
        """return the maximum value achievable."""
        return int(self.max_counter_label.text())

    @property
    def max_counter_label(self) -> QLabel | None:
        """return the max counter QLabel."""
        return self._max_counter_label

    @property
    def counter_label(self) -> QLabel | None:
        """return the counter QLabel."""
        return self._counter_label

    # ****** EVENT HANDLERS ****** #

    def _forward_pressed(self) -> None:
        """handle the clicking of the forward button."""
        if self.counter < self.max_counter:
            self.set_counter(self.counter + 1)
            self.move_forward.emit()
        self._forward_button.setEnabled(self.counter < self.max_counter)
        self._backward_button.setEnabled(self.counter > 1)

    def _backward_pressed(self) -> None:
        """handle the clicking of the backward button."""
        if self.counter > 0:
            self.set_counter(self.counter - 1)
            self.move_backward.emit()
        self._forward_button.setEnabled(self.counter < self.max_counter)
        self._backward_button.setEnabled(self.counter > 1)


class ResizableImage(QWidget):
    """create a widget accepting a 3D numpy array to be rendered as image."""

    # ****** SIGNALS ****** #

    _image_changed = None
    _mouse_pressed = None
    _mouse_released = None
    _mouse_doubleclick = None
    _mouse_moved = None
    _mouse_enter = None
    _mouse_leave = None
    _mouse_wheeling = None

    # ****** VARIABLES ****** #

    _ndarray = None
    _scale = None
    _image_coords = None
    _scroll = None
    _label = None

    # ****** CONSTRUCTOR ****** #

    def __init__(self) -> None:
        super().__init__()

        # setup the widget
        self._label = QLabel()
        self._scroll = QScrollArea()
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        policy = self._scroll.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Ignored)
        policy.setVerticalPolicy(QSizePolicy.Ignored)
        self._scroll.setSizePolicy(policy)
        self._scroll.setWidget(self._label)
        self._scroll.setWidgetResizable(True)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._scroll)
        self.setLayout(layout)

        # setup the event tracking
        self._label.setMouseTracking(True)
        self._label.installEventFilter(self)
        self._image_changed = Signal()
        self._mouse_pressed = Signal()
        self._mouse_released = Signal()
        self._mouse_doubleclick = Signal()
        self._mouse_moved = Signal()
        self._mouse_enter = Signal()
        self._mouse_leave = Signal()
        self._mouse_wheeling = Signal()

        # adjust the size
        self._scroll.resize(self._scroll.viewport().size())

    # ****** SETTERS ****** #

    def set_coords(self, pos: QPoint | None) -> None:
        """coords
        set the coordinates of the mouse pointer and update
        the data coords labels.

        Parameters
        ----------
        pos: QPoint | None
            the mouse position from which the coordinates are extracted.
        """
        if pos is None:
            self._image_coords = None
        else:
            check_type(pos, QPoint)
            mouse_x, mouse_y = pos.x(), pos.y()
            image_shape = self.ndarray.shape[:2][::-1]
            image_y = int(image_shape[1] * self._scale)
            space_y = self._scroll.height()
            y_off = (space_y - image_y) // 2
            if y_off > 0:
                self._image_coords = (mouse_x, mouse_y - y_off)
            else:
                self._image_coords = (mouse_x, mouse_y)

    def set_ndarray(self, ndarray: NDArray) -> None:
        """
        update the image with the provided ndarray.

        Parameters
        ----------
        ndarray: NDArray
            a 3D ndarray of the image to be rendered. The array must have
            dtype uint8 and should be designed according to the RGBA format.
        """
        check_type(ndarray, np.ndarray)
        assert ndarray.ndim == 3, "ndarray must be a 3D NDArray."
        assert ndarray.dtype == np.uint8, "ndarray must have dtype uint8."
        self._ndarray = ndarray

        # update the view as appropriate
        self.update_view()

    def set_scale(self, scale: float | int) -> None:
        """
        set the scaling factor between the raw image and the visualized one.

        Parameters
        ----------
        scale: int | float
            the scaling factor.
        """
        check_type(scale, (int, float))
        self._scale = scale

    # ****** GETTERS ****** #

    @property
    def mouse_enter(self) -> Signal | None:
        """return the mouse_enter signal"""
        return self._mouse_enter

    @property
    def mouse_leave(self) -> Signal | None:
        """return the mouse_leave signal"""
        return self._mouse_leave

    @property
    def mouse_pressed(self) -> Signal | None:
        """return the mouse_press signal"""
        return self._mouse_pressed

    @property
    def mouse_doubleclick(self) -> Signal | None:
        """return the mouse_doubleclick signal"""
        return self._mouse_doubleclick

    @property
    def mouse_moved(self) -> Signal | None:
        """return the mouse_moved signal"""
        return self._mouse_moved

    @property
    def mouse_released(self) -> Signal | None:
        """return the mouse_released signal"""
        return self._mouse_released

    @property
    def mouse_wheeling(self) -> Signal | None:
        """return the mouse_wheeling signal"""
        return self._mouse_wheeling

    @property
    def image_changed(self) -> Signal | None:
        """return the image_changed signal"""
        return self._image_changed

    @property
    def ndarray(self) -> NDArray | None:
        """return the ndarray defining the image."""
        return self._ndarray

    @property
    def scale(self) -> float | None:
        """
        return the scaling factor between the raw image and the visualized one.
        """
        return self._scale

    @property
    def image_coords(self) -> tuple[int, int] | None:
        """return the (x, y) coordinates of the mouse in pixels."""
        return self._image_coords

    @property
    def data_coords(self) -> tuple[int, int] | None:
        """return the (x, y) coordinates of the mouse in data units."""
        if self.image_coords is None or self.scale is None:
            return None
        x, y = tuple(int(i // self.scale) for i in self.image_coords)
        return x, y

    # ****** EVENT HANDLERS ****** #

    def eventFilter(self, obj: QObject, evt: QEvent) -> bool:
        """event filter handler"""
        is_under = isinstance(evt, QMouseEvent) and self.is_under_mouse(evt)

        # mouse wheeling
        if evt.type() == QEvent.Wheel:
            self._on_mouse_wheeling(evt)

        # mouse double-click
        if evt.type() == QEvent.MouseButtonDblClick and is_under:
            self._on_mouse_doubleclick(evt)
            return True

        # mouse press
        if (
            evt.type() == QMouseEvent.MouseButtonPress
            and evt.buttons() == Qt.LeftButton
            and is_under
        ):
            self._on_mouse_press(evt)
            return True

        # mouse release
        if evt.type() == QMouseEvent.MouseButtonRelease and is_under:
            self._on_mouse_release(evt)
            return True

        # mouse move
        if evt.type() == QMouseEvent.MouseMove:
            if is_under:
                self._on_mouse_move(evt)
            else:
                self._on_mouse_leave(evt)
            return True

        # enter
        if evt.type() == QEvent.Enter and is_under:
            self._on_mouse_enter(evt)
            return True

        # leave
        if evt.type() == QEvent.Leave:
            self._on_mouse_leave(evt)
            return True

        # return
        return super().eventFilter(obj, evt)

    def _on_mouse_press(self, event: QMouseEvent) -> None:
        """mouse press event."""
        self.set_coords(event.pos())
        self.mouse_pressed.emit(self.data_coords)

    def _on_mouse_doubleclick(self, event: QMouseEvent) -> None:
        """mouse double-click event."""
        self.set_coords(None)
        self.mouse_doubleclick.emit(self.data_coords)

    def _on_mouse_move(self, event: QMouseEvent) -> None:
        """mouse move event."""
        self.set_coords(event.pos())
        self.mouse_moved.emit(self.data_coords)

    def _on_mouse_release(self, event: QMouseEvent) -> None:
        """mouse release event."""
        self.set_coords(event.pos())
        self.mouse_released.emit(self.data_coords)

    def _on_mouse_enter(self, event: QMouseEvent) -> None:
        """image entering event"""
        self.set_coords(event.pos())
        self.mouse_enter.emit(self.data_coords)

    def _on_mouse_leave(self, event: QMouseEvent) -> None:
        """image leaving event"""
        self.set_coords(None)
        self.mouse_leave.emit(self.data_coords)

    def _on_mouse_wheeling(self, event: QEvent.Wheel) -> None:
        """mouse wheeling action"""
        self.set_coords(event.pos())
        self.mouse_wheeling.emit(event.angleDelta().y() / 15)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """image resizing event"""
        super().resizeEvent(event)
        self._scroll.viewport().resize(self._scroll.size())
        self.update_view()

    # ****** METHODS ****** #

    def is_under_mouse(self, event: QMouseEvent) -> bool:
        """return if the label is under the mouse."""
        if self.ndarray is None or self.scale is None:
            return False
        mouse = event.pos()
        mouse_x, mouse_y = mouse.x(), mouse.y()
        image_shape = self.ndarray.shape[:2][::-1]
        image_x, image_y = tuple(int(i * self._scale) for i in image_shape)
        space_y = self._label.height()
        y_off = (space_y - image_y) // 2
        x_on = mouse_x <= image_x
        if y_off > 0:
            y_on = mouse_y > y_off and mouse_y <= image_y + y_off
        else:
            y_on = mouse_y <= image_y
        return x_on and y_on

    def update_scale(self) -> None:
        """
        update the scaling factor between the ndarray and the
        available widget space.
        """
        hint = self._scroll.size()
        hint = (hint.width(), hint.height())
        raw = self.ndarray.shape[:2][::-1]
        self.set_scale(min(i / v for i, v in zip(hint, raw)))

    def update_view(self) -> None:
        """update the image with the provided ndarray."""
        self.update_scale()
        dsize = tuple(int(i * self.scale) for i in self.ndarray.shape[:2][::-1])
        interp = cv2.INTER_LINEAR
        img = cv2.resize(src=self.ndarray, dsize=dsize, interpolation=interp)
        self._label.setPixmap(ndarray2qpixmap(img))
        self.image_changed.emit(self)


class FileBrowseBar(QWidget):
    """
    widget allowing to set the path to a file on the system.

    Parameters
    ----------
    formats: Iterable[str]
        the list of accepted formats.
    """

    # ****** SIGNALS ****** #

    _text_changed = None

    # ****** VARIABLES ****** #

    _textfield = None
    _browse_button = None
    _formats = []
    _last_path = __file__.rsplit(os.path.sep, 1)[0]

    # ****** CONSTRUCTOR ****** #

    def __init__(self, formats: Iterable[str]) -> None:
        super().__init__()

        # setup the widget
        self._browse_button = QPushButton("BROWSE")
        self._browse_button.setFont(QFONT)
        size = self._browse_button.sizeHint()
        width = size.width()
        height = size.height()
        self._browse_button.setFixedSize(width + 5, height + 5)
        self._browse_button.clicked.connect(self._on_browse_press)
        self._textfield = QTextEdit("")
        self._textfield.setFont(QFONT)
        self._textfield.setEnabled(False)
        self._textfield.textChanged.connect(self._on_text_changed)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._textfield)
        layout.addWidget(self._browse_button)
        self.setLayout(layout)

        # setup the event tracking
        self._text_changed = Signal()
        self.setFixedHeight(height + 5)

        # setup the accepted formats
        if formats is not None:
            self.set_formats(formats)

    # ****** SETTERS ****** #

    def set_formats(self, formats) -> None:
        """
        Set the accepted file formats from the file browser

        Parameters
        ----------
        formats: Iterable[str]
            an iterable containing the accepted file formats.
        """
        self._formats = []

        # check the entries
        check_type(formats, Iterable)
        for ext in formats:
            check_type(ext, str)
            self._formats += [ext]

    # ****** GETTERS ****** #

    @property
    def text_changed(self) -> Signal | None:
        """return the file_changed signal"""
        return self._text_changed

    @property
    def text(self) -> Signal | None:
        """return the mouse_leave signal"""
        return self._textfield.toPlainText()

    @property
    def formats(self) -> list[str]:
        """return the list of accepted_formats."""
        return self._formats

    # ****** EVENT HANDLERS ****** #

    def _on_browse_press(self) -> None:
        """browse button press event."""
        file = QFileDialog.getOpenFileName(
            self,
            "Select File",
            self._last_path,
            "formats (" + " ".join([f"*.{i}" for i in self.formats]) + ")",
        )
        if file is not None:
            file = file[0].replace("/", os.path.sep)
            self._last_path = file.rsplit(os.path.sep, 1)[0]
            self._textfield.setText(file)

    def _on_text_changed(self) -> None:
        """text change event."""
        if not os.path.exists(self.text):
            QMessageBox.warning(
                self,
                "File not found",
                f"{self.text} not found.",
            )
        else:
            self.text_changed.emit(self.text)


class SaveBar(QWidget):
    """
    widget dedicated to highlight the progress of an action triggered by
    pressing one button.

    Parameters
    ----------
    minimum: int | float
        the minimum value of the progress bar.

    maximum: int | float
        the maximum value of the progress bar.

    step: int | float
        the step increment of the progress bar.
    """

    # ****** SIGNALS ****** #

    _started = None
    _completed = None

    # ****** VARIABLES ****** #

    _progress = None
    _button = None
    _step = None

    # ****** CONSTRUCTOR ****** #

    def __init__(
        self,
        minimum: int | float,
        maximum: int | float,
        step: int | float,
    ) -> None:
        super().__init__()

        # setup the widget
        self._button = QPushButton("SAVE")
        self._button.setFont(QFONT)
        size = self._button.sizeHint()
        width = size.width()
        height = size.height()
        self._button.setFixedSize(width + 5, height + 5)
        self._button.clicked.connect(self._on_button_press)
        self._progress = QProgressBar()
        self._progress.setFont(QFONT)
        self._progress.setTextVisible(False)
        self._progress.valueChanged.connect(self._on_value_changed)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._progress)
        layout.addWidget(self._button)
        self.setLayout(layout)
        self.setFixedHeight(height + 5)

        # setup the event tracking
        self._started = Signal()
        self._completed = Signal()

        # setters
        self.set_minimum(minimum)
        self.set_maximum(maximum)
        self.set_step(step)

    # ****** SETTERS ****** #

    def set_minimum(self, value: int | float) -> None:
        """
        set the minimum value of the progress bar.

        Parameters
        ----------
        minimum: int | float
            the minimum value to be set.
        """
        check_type(value, (int, float))
        self._progress.setMinimum(value)
        self.reset()

    def set_maximum(self, value: int | float) -> None:
        """
        set the maximum value of the progress bar.

        Parameters
        ----------
        maximum: int | float
            the maximum value to be set.
        """
        check_type(value, (int, float))
        self._progress.setMaximum(value)
        self.reset()

    def set_step(self, value: int | float) -> None:
        """
        set the step value of the progress bar.

        Parameters
        ----------
        maximum: int | float
            the maximum value to be set.
        """
        check_type(value, (int, float))
        assert value > 0, "step must be > 0"
        self._step = value
        self.reset()

    # ****** GETTERS ****** #

    @property
    def started(self) -> Signal | None:
        """return the start signal"""
        return self._started

    @property
    def completed(self) -> Signal | None:
        """return the completed signal"""
        return self._completed

    @property
    def minimum(self) -> int | float:
        """return the minimum value."""
        return self._progress.minimum()

    @property
    def maximum(self) -> int | float:
        """return the maximum value."""
        return self._progress.maximum()

    @property
    def step(self) -> int | float:
        """return the step value."""
        return self._step

    @property
    def value(self) -> int | float:
        """return the value of the progress bar"""
        return self._progress.value()

    # ****** EVENT HANDLERS ****** #

    def _on_button_press(self) -> None:
        """browse button press event."""
        self._button.setEnabled(False)
        self._progress.setTextVisible(True)
        self.started.emit()

    def _on_value_changed(self) -> None:
        """handle the change of the progress bar status"""
        if self.value == self.maximum:
            self.reset()
            self.completed.emit()

    # ****** METHODS ****** #

    def reset(self) -> None:
        """reset the values of the progress bar"""
        self._progress.setValue(self.minimum)
        self._button.setEnabled(True)
        self._progress.setTextVisible(False)

    def update(self) -> None:
        """
        update the progress
        """
        self._progress.setValue(min(self.value + self.step, self.maximum))
