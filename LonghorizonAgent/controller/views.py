from typing import Optional, Union, Tuple, Literal
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Literal, Union


class ActionResult(BaseModel):
    """Result of executing an action"""

    is_done: Optional[bool] = False
    success: Optional[bool] = None
    extracted_content: Optional[str] = None
    error: Optional[str] = None
    include_in_memory: bool = False  # whether to include in past messages as context or not


class DoneAction(BaseModel):
    text: str
    success: bool


PositionType = Union[int, Tuple[int, int], Tuple[int, float, float]]


class ClickAction(BaseModel):
    """Parameters for the click action."""
    position: PositionType = Field(
        ...,
        description="Click position: index (1-based) of a highlight box (int), exact coordinates (cx, cy), or a relative position within a highlight box (index, relative_x, relative_y where relative_x/y are floats from 0.0 to 1.0)."
    )


class LongPressAction(BaseModel):
    """Parameters for the long_press action."""
    position: PositionType = Field(
        ...,
        description="Long press position: index (1-based) of a highlight box (int), exact coordinates (cx, cy), or a relative position within a highlight box (index, relative_x, relative_y where relative_x/y are floats from 0.0 to 1.0)."
    )
    duration: Optional[float] = Field(
        1.0,
        description="Duration of the long press in seconds (default: 1.0)."
    )


class PressKeyAction(BaseModel):
    """Parameters for the press_key action."""
    key_name: Literal["home", "back", "recent"] = Field(
        ...,
        description="The special key to press ('home', 'back', or 'recent'). 'recent' usually triggers the app switch view."
    )


class InputTextAction(BaseModel):
    """Parameters for the input_text action."""
    position: Optional[PositionType] = Field(
        None,
        description="Input text position: index (1-based) of a highlight box (int), exact coordinates (cx, cy), or a relative position within a highlight box (index, relative_x, relative_y where relative_x/y are floats from 0.0 to 1.0)."
    )
    text: str = Field(..., description="The text to input into the currently focused field.")
    clear: Optional[bool] = Field(False,
                                  description="Whether to clear the field before inputting text (default: False).")


class SwipeAction(BaseModel):
    """Parameters for the swipe action."""
    start_position: PositionType = Field(
        ...,
        description="Swipe start position: index (1-based) of a highlight box (int), exact coordinates (cx, cy), or a relative position within a highlight box (index, relative_x, relative_y where relative_x/y are floats from 0.0 to 1.0)."
    )
    end_position: PositionType = Field(
        ...,
        description="Swipe end position: index (1-based) of a highlight box (int), exact coordinates (cx, cy), or a relative position within a highlight box (index, relative_x, relative_y where relative_x/y are floats from 0.0 to 1.0)."
    )


class DragAction(BaseModel):
    """Parameters for the drag action (similar to swipe)."""
    start_position: PositionType = Field(
        ...,
        description="Drag start position: index (1-based) of a highlight box (int), exact coordinates (cx, cy), or a relative position within a highlight box (index, relative_x, relative_y where relative_x/y are floats from 0.0 to 1.0)."
    )
    end_position: PositionType = Field(
        ...,
        description="Drag end position: index (1-based) of a highlight box (int), exact coordinates (cx, cy), or a relative position within a highlight box (index, relative_x, relative_y where relative_x/y are floats from 0.0 to 1.0)."
    )


class OpenAppAction(BaseModel):
    """Parameters for application launch operation"""
    app_identifier: str = Field(
        ...,
        description="Unique application identifier: package name (e.g., com.tencent.mm) or main activity name (e.g., com.tencent.mm.ui.LauncherUI)",
        example="com.tencent.mm"
    )
    launch_args: Optional[dict] = Field(
        None,
        description="Key-value pairs for launch arguments (e.g., {'scene':'1054'}) for special startup configurations"
    )
    verify_launch: Optional[bool] = Field(
        True,
        description="Verify successful application launch (default: True) by detecting homepage elements"
    )

class NavigateBackAction(BaseModel):
    """Parameters for system back navigation"""
    press_count: Optional[int] = Field(
        1,
        description="Consecutive back button presses (default: 1) for multi-level navigation",
        ge=1,
        le=5
    )
    target_page: Optional[str] = Field(
        None,
        description="Expected target page identifier (e.g., 'chat_list') for result validation"
    )
    timeout: Optional[float] = Field(
        3.0,
        description="Page load timeout in seconds (default: 3)",
        gt=0
    )

class WaitAction(BaseModel):
    duration: float = Field(
        ...,
        description="Waiting duration in seconds (supports fractional precision e.g., 0.5=500ms)",
        gt=0,
        le=60,
        example=2.0
    )
    condition: Optional[Literal["visible", "invisible", "clickable"]] = Field(
        None,
        description="Optional condition to end wait early: visible/invisible/clickable element state"
    )
    element_id: Optional[str] = Field(
        None,
        description="Associated element ID for conditional waiting (requires condition)"
    )
    polling_interval: Optional[float] = Field(
        0.5,
        description="Condition check polling interval in seconds (default: 0.5)",
        gt=0
    )

class ScrollAction(BaseModel):
    """Parameters for scroll operation"""
    direction: Literal["up", "down", "left", "right"] = Field(
        ...,
        description="Scroll direction: up/down/left/right"
    )
    distance_ratio: float = Field(
        0.8,
        description="Scroll distance ratio (0.1~1.0), 1.0 = full-screen scroll",
        gt=0.1,
        le=1.0
    )
    anchor_position: Optional[PositionType] = Field(
        None,
        description="Scroll anchor position for targeted area scrolling (e.g., chat list region)"
    )
    speed: Literal["slow", "normal", "fast"] = Field(
        "normal",
        description="Scroll speed: slow/normal/fast (affects operation duration)"
    )
    inertial: Optional[bool] = Field(
        True,
        description="Enable inertial scrolling (default: True) for realistic user behavior simulation"
    )
