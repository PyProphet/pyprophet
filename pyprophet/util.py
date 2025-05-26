import multiprocessing
from pathlib import Path
import sys
import ast
from importlib.metadata import version
from datetime import datetime
import platform
import numpy as np
import psutil
import click
from loguru import logger


class GlobalLogLevelGroup(click.Group):
    def invoke(self, ctx):
        log_level = ctx.params.get("log_level", "INFO").upper()
        header = setup_logger(log_level=log_level)
        ctx.obj = {"LOG_LEVEL": log_level, "LOG_HEADER": header}
        return super().invoke(ctx)


class AdvancedHelpGroup(click.Group):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add group-level helphelp
        self.params.append(
            click.Option(
                ["--helphelp"],
                is_flag=True,
                is_eager=True,
                expose_value=False,
                help="Show advanced help with all options.",
                callback=self.show_advanced_help,
            )
        )

    def show_advanced_help(self, ctx, param, value):
        if not value:
            return
        ctx.ensure_object(dict)
        ctx.obj["show_advanced"] = True
        click.echo(ctx.get_help(), color=ctx.color)
        ctx.exit()

    def command(self, *args, **kwargs):
        kwargs.setdefault("cls", AdvancedHelpCommand)
        return super().command(*args, **kwargs)


class AdvancedHelpCommand(click.Command):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add helphelp option to every command
        self.params.insert(
            0,
            click.Option(
                ["--helphelp"],
                is_flag=True,
                is_eager=True,
                expose_value=False,
                help="Show advanced help with all options.",
                callback=self._show_advanced_help,
            ),
        )

    def _show_advanced_help(self, ctx, param, value):
        if not value:
            return
        ctx.ensure_object(dict)
        ctx.obj["show_advanced"] = True
        click.echo(ctx.get_help(), color=ctx.color)
        ctx.exit()

    def format_options(self, ctx, formatter):
        ctx.ensure_object(dict)
        show_advanced = ctx.obj.get("show_advanced", False)

        opts = []
        for param in self.get_params(ctx):
            # Skip helphelp option itself
            if param.name == "helphelp":
                continue

            # Get normal help record first
            help_record = param.get_help_record(ctx)

            # For hidden options when in advanced mode
            if show_advanced and getattr(param, "hidden", False):
                # Reconstruct the full option declaration
                if param.is_flag:
                    if param.secondary_opts:
                        # For --flag/--no-flag style boolean options
                        opt_decl = " / ".join(sorted(param.opts + param.secondary_opts))
                    else:
                        # For simple flags
                        opt_decl = ", ".join(sorted(param.opts))
                else:
                    # For regular options
                    opt_decl = ", ".join(sorted(param.opts))

                # Add type information
                if param.metavar:
                    opt_decl += f" {param.metavar}"
                elif isinstance(param.type, click.Tuple):
                    # Handle tuple types (like pi0_lambda)
                    types = " ".join([t.name.upper() for t in param.type.types])
                    opt_decl += f" <{types}>..."
                elif isinstance(param.type, click.Choice):
                    opt_decl += f' [{"/".join(param.type.choices)}]'
                elif hasattr(param.type, "name"):
                    if param.type.name in ["integer", "float", "boolean"]:
                        opt_decl += f" {param.type.name.upper()}"
                    elif param.type.name == "text":
                        opt_decl += " TEXT"

                # Include default if shown
                default_part = ""
                if param.show_default:
                    if isinstance(param.default, (list, tuple)):
                        default = str(list(param.default))
                    else:
                        default = (
                            str(param.default) if param.default is not None else ""
                        )
                    if default:
                        default_part = f"  [default: {default}]"
                    elif param.required:
                        default_part = "  [required]"

                # Create the full help record
                help_record = (opt_decl, (param.help or "") + default_part)

            if help_record is not None:
                opts.append(help_record)

        if not show_advanced:
            helphelp_param = next(p for p in self.params if p.name == "helphelp")
            opts.append(("--helphelp", "Show advanced help with all options"))

        if opts:
            with formatter.section("Options"):
                formatter.write_dl(opts, col_max=35, col_spacing=6)


class CombinedGroup(GlobalLogLevelGroup, AdvancedHelpGroup):
    """Combines both functionalities"""

    def invoke(self, ctx):
        # Initialize context
        ctx.ensure_object(dict)

        # First check for helphelp flag
        if "--helphelp" in ctx.args:
            ctx.obj["show_advanced"] = True

        # Proceed with normal invocation
        return super().invoke(ctx)


# https://stackoverflow.com/a/47730333
class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        if not isinstance(value, str):  # required for Click>=8.0.0
            return value
        try:
            return ast.literal_eval(value)
        except Exception:
            raise click.BadParameter(value)


_LOGGER_INITIALIZED = False  # Module-level flag


def get_system_info():
    """Get OS, Python version, CPU, and RAM info."""
    return (
        f"OS: {platform.system()} {platform.release()} | "
        f"Python: {platform.python_version()} | "
        f"CPU: {psutil.cpu_count()} cores | "
        f"RAM: {psutil.virtual_memory().total / (1024 ** 3):.1f} GB"
    )


def get_execution_context():
    """Get the exact CLI command used (works with Click subcommands)."""
    return " ".join([sys.executable] + sys.argv)


def setup_logger(log_level):
    def formatter(record):
        # Format with module, function, and line number
        # mod_func_line = f"{record['module']}::{record['function']}:{record['line']}"
        # return (
        #     f"[ <green>{record['time']:YYYY-MM-DD at HH:mm:ss}</green> | "
        #     f"<level>{record['level']: <7}</level> | "
        #     f"{mod_func_line: <37} ] "
        #     f"<level>{record['message']}</level>\n"
        # )
        mod_func_line = f"{record['module']}::{record['line']}"
        return (
            f"[ <green>{record['time']:YYYY-MM-DD at HH:mm:ss}</green> | "
            f"<level>{record['level']: <7}</level> | "
            f"{mod_func_line: <27} ] "
            f"<level>{record['message']}</level>\n"
        )

    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        return

    # simple logger for header info
    header_logger = logger.bind(simple=True)
    header_logger.remove()
    header_logger.add(
        sys.stdout,
        format="{message}",
        level=log_level,
        filter=lambda record: "simple" in record["extra"],
    )

    # Log header info
    header = (
        f"PyProphet v{version('pyprophet')}\n"
        f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"System: {get_system_info()}\n"
        f"Command: {get_execution_context()}\n"
    )
    header_logger.info(header)

    # Main console logger
    # logger.remove()  # Remove default logger
    logger.add(sys.stdout, colorize=True, format=formatter, level=log_level)

    _LOGGER_INITIALIZED = True

    return header


def write_logfile(log_level, log_file, log_header=None):
    def formatter(record):
        # Format with module, function, and line number
        mod_func_line = f"{record['module']}::{record['function']}:{record['line']}"
        return (
            f"[ <green>{record['time']:YYYY-MM-DD at HH:mm:ss}</green> | "
            f"<level>{record['level']: <7}</level> | "
            f"{mod_func_line: <45} ] "
            f"<level>{record['message']}</level>\n"
        )

    log_file = Path(log_file)
    if log_file.exists():
        log_file.unlink()

    if log_header:
        with log_file.open("w") as f:
            f.write(log_header)

    logger.add(
        log_file,
        colorize=False,
        format=formatter,
        level=log_level,
        rotation="1000 MB",
    )


# Parameter transformation functions
def transform_pi0_lambda(ctx, param, value):
    if value[1] == 0 and value[2] == 0:
        pi0_lambda = value[0]
    elif 0 <= value[0] < 1 and value[0] <= value[1] <= 1 and 0 < value[2] < 1:
        pi0_lambda = np.arange(value[0], value[1], value[2])
    else:
        raise click.ClickException(
            "Wrong input values for pi0_lambda. pi0_lambda must be within [0,1)."
        )
    return pi0_lambda


def transform_threads(ctx, param, value):
    if value == -1:
        value = multiprocessing.cpu_count()
    return value


def transform_subsample_ratio(ctx, param, value):
    if value < 0 or value > 1:
        raise click.ClickException(
            "Wrong input values for subsample_ratio. subsample_ratio must be within [0,1]."
        )
    return value
