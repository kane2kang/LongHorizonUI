import pdb
import time
import logging
from typing import Optional

from .registry.service import Registry
from .registry.views import ActionModel

from .views import (
    DoneAction,
    ActionResult,

)

from ..system.system_context import SystemContext

logger = logging.getLogger(__name__)


class BaseController:
    def __init__(
            self,
            exclude_actions: list[str] = [],
            highlight_action: bool = True
    ):
        self.registry = Registry(exclude_actions)
        self.highlight_action = highlight_action
        self.register_common_actions()
        self.register_custom_actions()

    def register_common_actions(self):
        """
        公用的actions
        :return:
        """

        @self.registry.action(
            'Complete task - with return text and if the task is finished (success=True) or not yet completly finished (success=False), because last step is reached',
            param_model=DoneAction,
        )
        def done(params: DoneAction):
            return ActionResult(is_done=True, success=params.success, extracted_content=params.text)

        @self.registry.action(
            'Wait for 3 seconds. Usually used when waiting for a page to load or download.')
        def wait():
            seconds = 3
            msg = f'Waiting for {seconds} seconds'
            logger.info(msg)
            time.sleep(seconds)
            return ActionResult(extracted_content=msg, include_in_memory=True)

    def register_custom_actions(self):
        """
        针对不同平台定制化的action
        :return:
        """
        pass

    def act(
            self,
            action: ActionModel,
            system_context: Optional[SystemContext] = None,
    ) -> ActionResult:
        """Execute an action"""

        try:
            for action_name, params in action.model_dump(exclude_unset=True).items():
                if params is not None:
                    result = self.registry.execute_action(
                        action_name,
                        params,
                        system_context=system_context
                    )
                    if isinstance(result, str):
                        return ActionResult(extracted_content=result)
                    elif isinstance(result, ActionResult):
                        return result
                    elif result is None:
                        return ActionResult()
                    else:
                        raise ValueError(f'Invalid action result type: {type(result)} of {result}')
            return ActionResult()
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e
