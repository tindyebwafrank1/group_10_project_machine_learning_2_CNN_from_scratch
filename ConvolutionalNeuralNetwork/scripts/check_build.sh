
set -euo pipefail

INC="${INC:--I. -Iinclude -Isrc}"
TESTS="${TESTS:-test_preprocess test_stride test_fully_connected}"
DEMOS="${DEMOS:-stride_demo fully_connected_demo}"
OUT=build-check
STRICT="-O3 -Wall -Wextra -pedantic -std=c++17 -Werror"

rm -rf "$OUT" && mkdir -p "$OUT"

for t in $TESTS; do
  [ -f "tests/$t.cpp" ] || { echo "MISSING tests/$t.cpp"; exit 1; }
  echo "== $t (strict release flags)"
  g++ $STRICT $INC "tests/$t.cpp" -o "$OUT/$t"
  "./$OUT/$t"
done

for d in $DEMOS; do
  [ -f "examples/$d.cpp" ] || { echo "MISSING examples/$d.cpp"; exit 1; }
  echo "== $d (demo)"
  g++ $STRICT $INC "examples/$d.cpp" -o "$OUT/$d"
  "./$OUT/$d" > "$OUT/$d.out"
  echo "   ran OK, output saved to $OUT/$d.out"
done

for t in $TESTS; do
  echo "== $t (address + undefined-behaviour sanitizers)"
  g++ -g -O1 -Wall -Wextra -std=c++17 -fsanitize=address,undefined -fno-omit-frame-pointer \
      $INC "tests/$t.cpp" -o "$OUT/$t-san"
  "./$OUT/$t-san"
done

if command -v valgrind >/dev/null 2>&1; then
  for t in $TESTS; do
    echo "== $t (valgrind)"
    g++ -g -O0 -std=c++17 $INC "tests/$t.cpp" -o "$OUT/$t-vg"
    valgrind --leak-check=full --error-exitcode=1 "./$OUT/$t-vg"
  done
else
  echo "== valgrind not installed: skipped (sanitizer run above still checks leaks and bounds)"
fi

echo "BUILD CHECK PASSED"