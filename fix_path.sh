#!/bin/bash

echo "Replacing $1 in mlruns/$3 with $2 ; Confirm? (y/N)"
read answer
if [ "$answer" != "y" ]
then
    exit 0
fi

find "mlruns/$3" -type f -exec sed -i -e "s|$1|$2|g" {} \;