#
# setup.py
#
# Thanks to
#    https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/
#    https://python-packaging.readthedocs.io/en/latest/minimal.html

import codecs
import os
import re

from setuptools import setup

#-----------------------------------------------------------------------------------------------------------------------

DIST_NAME = 'rgatkinson'
AUTHOR = 'Robert Atkinson'
EMAIL = 'bob@theatkinsons.org'
KEYWORDS = ["robots", "protocols", "synbio", "pcr", "automation", "lab", "opentrons"]
DESCRIPTION = "Capability enhancements to the Opentrons API"
VERSION = '1.0.16'
PACKAGES = [DIST_NAME]
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: Implementation",
]
INSTALL_REQUIRES = []

#-----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    setup(
        python_requires='>=3.6',
        name=DIST_NAME,
        description=DESCRIPTION,
        version=VERSION,
        packages=PACKAGES,
        zip_safe=False,
        author=AUTHOR,
        author_email=EMAIL,
        maintainer=AUTHOR,
        maintainer_email=EMAIL,
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        url='https://github.com/rgatkinson/OpentronsProtocols',
        package_dir={'': 'src'},
        project_urls={
            'Source Code On Github': "https://github.com/rgatkinson/OpentronsProtocols"
        }
    )

