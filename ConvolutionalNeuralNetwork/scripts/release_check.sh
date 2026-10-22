
set -uo pipefail

MAIN="${MAIN:-main.cpp}"
INC="${INC:--I. -Iinclude -Isrc}"
OUT=release-check-build
TEST_TIMEOUT="${TEST_TIMEOUT:-120}"
APP_TIMEOUT="${APP_TIMEOUT:-600}"
PASS=0
FAIL=0
SKIP=0

ok()   { echo "  PASS  $1"; PASS=$((PASS + 1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL + 1)); }
skip() { echo "  SKIP  $1"; SKIP=$((SKIP + 1)); }

rm -rf "$OUT" && mkdir -p "$OUT"

# Safely check for tensor.cpp to avoid breaking builds if it doesn't exist
TENSOR_SRC=""
if [ -f "src/tensor.cpp" ]; then
  TENSOR_SRC="src/tensor.cpp"
fi

echo "== 1. Official release build"
if [ -f "$MAIN" ]; then
  if g++ -O3 -Wall -Wextra -std=c++17 $INC "$MAIN" $TENSOR_SRC -o "$OUT/cnn_app" 2> "$OUT/main_build.log"; then
    if [ -s "$OUT/main_build.log" ]; then
      bad "$MAIN builds but prints warnings (see $OUT/main_build.log)"
      head -n 15 "$OUT/main_build.log"
    else
      ok "$MAIN builds with no warnings"
    fi
    if [ "${RUN_APP:-0}" = "1" ]; then
      if timeout "$APP_TIMEOUT" "./$OUT/cnn_app" > "$OUT/cnn_app.out" 2>&1; then
        ok "cnn_app ran and exited with code 0 (output in $OUT/cnn_app.out)"
      else
        bad "cnn_app failed or timed out (see $OUT/cnn_app.out)"
      fi
    else
      skip "running cnn_app (set RUN_APP=1 to run it)"
    fi
  else
    bad "$MAIN does not build (see $OUT/main_build.log)"
    head -n 15 "$OUT/main_build.log"
  fi
else
  bad "$MAIN not found (set MAIN=path/to/main.cpp)"
fi

echo "== 2. Headers compile on their own"
HEADERS=$(find include -type f \( -name '*.h' -o -name '*.hpp' \) 2>/dev/null | sort)
if [ -z "$HEADERS" ]; then
  skip "no headers found under include/"
else
  while IFS= read -r h; do
    if g++ -std=c++17 -Wall -Wextra -fsyntax-only $INC -x c++ "$h" $TENSOR_SRC 2> "$OUT/header.log"; then
      ok "$h"
    else
      bad "$h does not compile alone (missing include?)"
      head -n 6 "$OUT/header.log"
    fi
  done <<< "$HEADERS"
fi

echo "== 3. No global 'using namespace std;' in headers (heuristic: counts braces, ignores // comments)"
if [ -z "$HEADERS" ]; then
  skip "no headers"
else
  found=0
  while IFS= read -r h; do
    hits=$(awk '
      { line = $0; sub(/\/\/.*/, "", line)
        if (depth == 0 && line ~ /^[[:space:]]*using[[:space:]]+namespace[[:space:]]+std[[:space:]]*;/) print FILENAME ":" FNR
        n = gsub(/\{/, "{", line); m = gsub(/\}/, "}", line); depth += n - m }' "$h")
    if [ -n "$hits" ]; then
      bad "global using namespace std in $hits"
      found=1
    fi
  done <<< "$HEADERS"
  [ "$found" -eq 0 ] && ok "no header opens std at global scope"
fi

echo "== 4. Tests"
SRCS=$(grep -L "int main" src/*.cpp 2>/dev/null || true)
run_test() {  # $1 = test file, $2 = extra compile flags, $3 = label suffix
  local t="$1" flags="$2" label="$3"
  local name; name=$(basename "$t" .cpp)
  local exe="$OUT/${name}${label}"
  if g++ $flags $INC "$t" $TENSOR_SRC -o "$exe" 2> "$OUT/${name}${label}.build.log"; then
    :
  elif grep -q "undefined reference" "$OUT/${name}${label}.build.log" && [ -n "$SRCS" ] &&
       g++ $flags $INC "$t" $SRCS $TENSOR_SRC -o "$exe" 2> "$OUT/${name}${label}.build.log"; then
    :
  else
    bad "$name${label}: does not build (see $OUT/${name}${label}.build.log)"
    head -n 8 "$OUT/${name}${label}.build.log"
    return
  fi
  if [ -s "$OUT/${name}${label}.build.log" ] && [ -z "$label" ]; then
    bad "$name builds but prints warnings (see $OUT/${name}${label}.build.log)"
    return
  fi
  if timeout "$TEST_TIMEOUT" "./$exe" > "$OUT/${name}${label}.out" 2>&1; then
    ok "$name${label}"
  else
    bad "$name${label}: failed or timed out (see $OUT/${name}${label}.out)"
    tail -n 8 "$OUT/${name}${label}.out"
  fi
}

TESTFILES=$(ls tests/*.cpp 2>/dev/null | sort)
if [ -z "$TESTFILES" ]; then
  skip "no tests found in tests/"
else
  while IFS= read -r t; do
    run_test "$t" "-O3 -Wall -Wextra -std=c++17" ""
  done <<< "$TESTFILES"
fi

echo "== 5. Sanitizers (address + undefined behaviour)"
if [ "${SAN:-0}" = "1" ] && [ -n "$TESTFILES" ]; then
  while IFS= read -r t; do
    run_test "$t" "-g -O1 -std=c++17 -fsanitize=address,undefined -fno-omit-frame-pointer" "-san"
  done <<< "$TESTFILES"
else
  skip "set SAN=1 to run every test under the sanitizers"
fi

echo "== 6. Repository basics"
[ -f README.md ] && ok "README.md exists" || bad "README.md is missing"
[ -f CMakeLists.txt ] && ok "CMakeLists.txt exists" || bad "CMakeLists.txt is missing"
[ -f .gitignore ] && ok ".gitignore exists" || bad ".gitignore is missing"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  tracked=$(git ls-files | grep -E '(^|/)(build|build-check|release-check-build)/|\.(o|obj|exe|out)$' || true)
  if [ -n "$tracked" ]; then
    bad "build files are tracked by git:"
    echo "$tracked" | head -n 10
  else
    ok "no build files tracked by git"
  fi
  if [ -n "$(git status --porcelain)" ]; then
    bad "working tree has uncommitted or untracked changes (commit or ignore them before release)"
    git status --short | head -n 10
  else
    ok "working tree is clean"
  fi
else
  skip "not a git repository"
fi

echo
echo "SUMMARY: $PASS passed, $FAIL failed, $SKIP skipped"
if [ "$FAIL" -gt 0 ]; then
  echo "RELEASE CHECK FAILED"
  exit 1
fi
echo "RELEASE CHECK PASSED"
