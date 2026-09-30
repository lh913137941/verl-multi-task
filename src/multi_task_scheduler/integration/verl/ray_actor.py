"""Isolate the Ray class-unwrapping mechanism already used by verl."""


def unwrap_native_actor_class(actor_class: object) -> type:
    """Return the real Python base, never an ActorHandle or a fake class.

    verl/single_controller/ray/base.py::_unwrap_ray_remote uses the same Ray
    attribute. A changed Ray representation must fail before actor construction.
    """
    native_class = getattr(actor_class, "__ray_actor_class__", None)
    if not isinstance(native_class, type):
        raise TypeError("Expected a Ray ActorClass exposing __ray_actor_class__; check verl/Ray compatibility")
    return native_class
