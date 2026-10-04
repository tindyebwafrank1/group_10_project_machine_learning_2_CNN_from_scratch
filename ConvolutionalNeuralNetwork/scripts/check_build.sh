#!/usr/bin/env bash
# Usage: INC="-Iinclude" ./scripts/check_build.sh   (default INC searches ., include, src)
set -euo pipefail
INC="${INC:--I. -Iinclude -Isrc}"
OUT=build-check
rm -rf "$OUT" && mkdir -p "$OUT"
STRICT="-O3 -Wall -Wextra -std=c++17 -Werror"
TESTS="test_tensor_contract test_preprocess"

for t in $TESTS; do
  echo "== $t (strict release flags)"
  g++ $STRICT $INC "tests/$t.cpp" -o "$OUT/$t"
  "./$OUT/$t"
done

echo "== demo"
g++ $STRICT $INC examples/preprocess_demo.cpp -o "$OUT/preprocess_demo"
"./$OUT/preprocess_demo"

for t in $TESTS; do
  echo "== $t (address + undefined-behaviour sanitizers)"
  g++ -g -O1 -Wall -Wextra -std=c++17 -fsanitize=address,undefined -fno-omit-frame-pointer \
      $INC "tests/$t.cpp" -o "$OUT/$t-san"
  "./$OUT/$t-san"
done

if command -v valgrind >/dev/null 2>&1; then
  echo "== valgrind"
  g++ -g -O0 -std=c++17 $INC tests/test_preprocess.cpp -o "$OUT/test_preprocess-vg"
  valgrind --leak-check=full --error-exitcode=1 "./$OUT/test_preprocess-vg"
fi
echo "BUILD CHECK PASSED"