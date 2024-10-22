# coding: utf-8

"""
Helpful utilities and functions.
"""

from __future__ import annotations

import os
import sys
import copy
import random
import inspect
import contextlib

from sync._types import Any, GenericAlias, Callable


colors = {
    "default": 39,
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "light_gray": 37,
    "dark_gray": 90,
    "light_red": 91,
    "light_green": 92,
    "light_yellow": 93,
    "light_blue": 94,
    "light_magenta": 95,
    "light_cyan": 96,
    "white": 97,
}

backgrounds = {
    "default": 49,
    "black": 40,
    "red": 41,
    "green": 42,
    "yellow": 43,
    "blue": 44,
    "magenta": 45,
    "cyan": 46,
    "light_gray": 47,
    "dark_gray": 100,
    "light_red": 101,
    "light_green": 102,
    "light_yellow": 103,
    "light_blue": 104,
    "light_magenta": 105,
    "light_cyan": 106,
    "white": 107,
}

styles = {
    "default": 0,
    "bright": 1,
    "dim": 2,
    "underlined": 4,
    "blink": 5,
    "inverted": 7,
    "hidden": 8,
}


def colored(
    msg: Any,
    color: str = "default",
    background: str = "default",
    style: str | list[str] = "default",
) -> str:
    """
    Return the colored version of a string *msg*. For *color*, *background* and *style* options, see
    https://misc.flogisoft.com/bash/tip_colors_and_formatting. They can also be explicitely set to
    ``"random"`` to get a random value. *msg* is returned unchanged in case the output is a tty.
    """
    _msg = str(msg)
    # check for tty
    tty = False
    try:
        tty = os.isatty(sys.stdout.fileno())
    except:
        pass
    if not tty:
        return _msg

    if color == "random":
        _color = random.choice(list(colors.values()))
    else:
        _color = colors.get(color, colors["default"])

    if background == "random":
        _background = random.choice(list(backgrounds.values()))
    else:
        _background = backgrounds.get(background, backgrounds["default"])

    style_values = list(styles.values())
    _style = ";".join(
        str(random.choice(style_values) if s == "random" else styles.get(s, styles["default"]))
        for s in ([style] if isinstance(style, str) else style)
    )

    return f"\033[{_style};{_background};{_color}m{_msg}\033[0m"


def is_pattern(s: str) -> bool:
    return "?" in s or "*" in s


def print_usage(funcs: list[Callable], margin: bool = True) -> None:
    if margin:
        print(f"\n{' Usage '.center(100, '-')}")

    for func in funcs:
        print("")
        sig = colored(str(inspect.signature(func)), style="bright")
        sig = sig.replace("->", colored("->", "magenta"))
        sig = sig.replace("'", "")
        print(f"{colored(func.__name__, 'green')}{sig}")
        if func.__doc__:
            doc = func.__doc__.strip().replace(12 * " ", 4 * " ").replace(8 * " ", 4 * " ")
            print(f"    {doc}")

    if margin:
        print(f"\n{100 * '-'}\n")


@contextlib.contextmanager
def change_stdout(f):
    orig_stdout = sys.stdout
    sys.stdout = f
    try:
        yield
    finally:
        sys.stdout = orig_stdout


class DotDict(dict):
    """
    Dictionary subclass that provides read access for items via attributes by implementing
    ``__getattr__``. In case a item is accessed via attribute and it does not exist, an
    *AttriuteError* is raised rather than a *KeyError*. Example:

    .. code-block:: python

        d = DotDict()
        d["foo"] = 1

        print(d["foo"])
        # => 1

        print(d.foo)
        # => 1

        print(d["bar"])
        # => KeyError

        print(d.bar)
        # => AttributeError
    """

    def __class_getitem__(cls, types: tuple[type, type]) -> GenericAlias:
        return GenericAlias(cls, types)  # type: ignore[call-overload, return-value]

    @classmethod
    def wrap(cls, *args, **kwargs) -> DotDict:
        """
        Takes a dictionary *d* and recursively replaces it and all other nested dictionary types
        with :py:class:`DotDict`'s for deep attribute-style access.
        """
        wrap = lambda d: cls((k, wrap(v)) for k, v in d.items()) if isinstance(d, dict) else d  # type: ignore # noqa
        return wrap(dict(*args, **kwargs))

    def __getattr__(self, attr: str) -> Any:
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def __setattr__(self, attr: str, value: Any) -> None:
        self[attr] = value

    def copy(self) -> DotDict:
        """"""
        return copy.deepcopy(self)
