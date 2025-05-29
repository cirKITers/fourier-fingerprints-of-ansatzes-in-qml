#!/bin/bash

echo "Replacing $1 in mlruns/ with $2 ; Confirm? (y/N)"
read answer
if [ "$answer" != "y" ]
then
    exit 0
fi

find mlruns -type f -exec sed -i -e "s|$1|$2|g" {} \;