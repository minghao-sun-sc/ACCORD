"""Allow ``python -m accord``."""

import sys

from .cli import main

sys.exit(main())
