#!/bin/bash
# Generate html documentation for a single python module
# https://stackoverflow.com/questions/10692228/simplest-way-to-run-sphinx-on-one-python-file

#PACKAGE=${PWD##*/}
PACKAGE="paragami"
#MODULE="$1"
MODULE='sensitivity_lib\\.py'
#MODULE_NAME=${MODULE%\\\\.py}
PACKAGE_DIR="/home/rgiordan/Documents/git_repos/paragami/paragami"
DOCDIR="/home/rgiordan/Documents/git_repos/paragami/docs/tmp_build"
CONFLOC="/home/rgiordan/Documents/git_repos/paragami/docs/source/"

# mkdir -p .tmpdocs
# rm -rf .tmpdocs/*
# Exclude all directories
# Exclude all other modules (apidoc crashes if __init__.py is excluded)
# sphinx-apidoc \
#     -f -e --module-first --no-toc -o $DOCDIR "$PACKAGE_DIR" \
#     $(find "$PACKAGE_DIR" -maxdepth 1 -mindepth 1 -type d) \
#     $(find "$PACKAGE_DIR" -maxdepth 1 -regextype posix-egrep \
#         ! -regex ".*/$MODULE|.*/__init__.py" -type f)
sphinx-apidoc \
    -f -e --module-first --no-toc -o $DOCDIR "$PACKAGE_DIR" \
    $(find "$PACKAGE_DIR" -maxdepth 1 -mindepth 1 -type d) \
    $(find "$PACKAGE_DIR" -maxdepth 1 -regextype posix-egrep \
        ! -regex ".*/$MODULE|.*/__init__.py" -type f)
ls $DOCDIR


# This doesn't seem to work.
MODULE='sensitivity_lib'
find "$PACKAGE_DIR" -maxdepth 1 -regextype posix-egrep ! -regex $MODULE -type f
find "$PACKAGE_DIR" -maxdepth 1 -regextype posix-egrep ! -regex $MODULE"|.*/__init__.py" -type f
find "$PACKAGE_DIR" -maxdepth 1 -regextype posix-egrep ! -regex $MODULE -type f
find "$PACKAGE_DIR" -maxdepth 1 -regextype posix-egrep ! -regex ".*/__init__.py" -type f
#rm $DOCDIR/$PACKAGE.rst
# build crashes if index.rst does not exist
touch $DOCDIR/index.rst
sphinx-build -b html -c $CONFLOC  -d $DOCDIR \
    $DOCDIR $DOCDIR/output $DOCDIR/*.rst

echo "**** HTML-documentation for $MODULE is available in $DOCDIR/$PACKAGE.$MODULE_NAME.html"
