#!/usr/bin/env bash
# Stamp the workspace version from a release tag.
#
#   scripts/ci/stamp_version.sh <version>       # e.g. 0.2.1, 0.2.1-beta.2
#
# Rewrites `[workspace.package] version` in the root Cargo.toml. Every member
# inherits it through `version.workspace = true`.
# It also pins the `nexar` workspace dependency with `=`, because a prerelease
# satisfies no caret requirement on its base version.
#
# Only a prerelease tag changes anything. release-validate.yml checks the tag's
# base version against the manifest, so `v0.2.1` is a no-op and
# `v0.2.1-beta.2` appends the suffix. A re-run is idempotent.

set -euo pipefail

VERSION="${1:?usage: stamp_version.sh <version>}"

CURRENT=$(cargo metadata --no-deps --format-version=1 \
    | jq -r '.packages[] | select(.name == "nexar") | .version')

if [[ "$VERSION" == "$CURRENT" ]]; then
    echo "Version already $VERSION, nothing to stamp."
    exit 0
fi

perl -i -pe 'if (/^\[/) { $in = /^\[workspace\.package\]/ } if ($in && !$done && s/^version = ".*"/version = "'"$VERSION"'"/) { $done = 1 }' Cargo.toml

# A prerelease satisfies no caret requirement, so the path dependency pins exactly.
perl -i -pe 's/^(nexar = \{ version = )"[^"]*"/$1"='"$VERSION"'"/' Cargo.toml

echo "Stamped workspace: $CURRENT -> $VERSION"
