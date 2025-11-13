#!/bin/bash

set -e

export PYTHONPATH=$(pwd)


case $1 in 
    record)
    python -B "$PYTHONPATH/scripts/app.py" 
    ;;

    train|validate|test|model|embedding|clr|spec)
    python -B "$PYTHONPATH/scripts/pipeline.py" "$1"
    ;;

    *)
    echo "❌ Invalid option!"
    echo "✅ Usage: $0 [record|train|validate|test|model|embedding|clr]"
    exit 1
    ;;
esac