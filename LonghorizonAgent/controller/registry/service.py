import asyncio
from inspect import iscoroutinefunction, signature
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field, create_model

from .views import ActionRegistry, ActionModel, RegisteredAction
from ...system.system_context import SystemContext


class Registry:
    """Service for registering and managing actions"""

    def __init__(self, exclude_actions: list[str] = []):
        self.registry = ActionRegistry()
        self.exclude_actions = exclude_actions

    def _create_param_model(self, function: Callable) -> Type[BaseModel]:
        """Creates a Pydantic model from function signature"""
        sig = signature(function)
        params = {
            name: (param.annotation, ... if param.default == param.empty else param.default)
            for name, param in sig.parameters.items() if name != 'system_context'
        }
        # TODO: make the types here work
        return create_model(
            f'{function.__name__}_parameters',
            __base__=ActionModel,
            **params,  # type: ignore
        )

    def action(
            self,
            description: str,
            param_model: Optional[Type[BaseModel]] = None,
    ):
        """Decorator for registering actions"""

        def decorator(func: Callable):
            # Skip registration if action is in exclude_actions
            if func.__name__ in self.exclude_actions:
                return func

            # Create param model from function if not provided
            actual_param_model = param_model or self._create_param_model(func)

            action = RegisteredAction(
                name=func.__name__,
                description=description,
                function=func,
                param_model=actual_param_model,
            )
            self.registry.actions[func.__name__] = action
            return func

        return decorator

    def execute_action(
            self,
            action_name: str,
            params: dict,
            system_context: Optional[SystemContext] = None,
    ) -> Any:
        """Execute a registered action"""
        if action_name not in self.registry.actions:
            raise ValueError(f'Action {action_name} not found')

        action = self.registry.actions[action_name]
        try:
            # Create the validated Pydantic model
            validated_params = action.param_model(**params)

            # Check if the first parameter is a Pydantic model
            sig = signature(action.function)
            parameters = list(sig.parameters.values())
            is_pydantic = parameters and issubclass(parameters[0].annotation, BaseModel)
            parameter_names = [param.name for param in parameters]

            if 'system_context' in parameter_names and not system_context:
                raise ValueError(f'Action {action_name} requires system context but none provided.')

            # Prepare arguments based on parameter type
            extra_args = {}
            if 'system_context' in parameter_names:
                extra_args['system_context'] = system_context
            if is_pydantic:
                return action.function(validated_params, **extra_args)
            return action.function(**validated_params.model_dump(), **extra_args)

        except Exception as e:
            raise RuntimeError(f'Error executing action {action_name}: {str(e)}') from e

    def create_action_model(self) -> Type[ActionModel]:
        """Creates a Pydantic model from registered actions"""
        fields = {
            name: (
                Optional[action.param_model],
                Field(default=None, description=action.description),
            )
            for name, action in self.registry.actions.items()
        }

        return create_model('ActionModel', __base__=ActionModel, **fields)  # type:ignore

    def get_prompt_description(self) -> str:
        """Get a description of all actions for the prompt"""
        return self.registry.get_prompt_description()
