
#include <iostream>
#include <stdexcept>
#include <cmath>
#include "tensor.hpp"

using namespace std;

int main()
{
    int failures = 0;

    // Test 1: Constructor and size.
    Tensor t(1, 1, 3, 3);

    if (t.size() != 9)
    {
        cerr << "FAILED: Constructor and size\n";
        failures++;
    }

    // Test 2: New tensors should contain zeros.
    if (t(0, 0, 0, 0) != 0.0)
    {
        cerr << "FAILED: Initial zero values\n";
        failures++;
    }

    // Test 3: Write and read an element.
    t(0, 0, 1, 1) = 5.0;

    if (t(0, 0, 1, 1) != 5.0)
    {
        cerr << "FAILED: Element access\n";
        failures++;
    }

    // Test 4: Fill.
    t.fill(2.0);

    if (t(0, 0, 0, 0) != 2.0 ||
        t(0, 0, 2, 2) != 2.0)
    {
        cerr << "FAILED: Fill\n";
        failures++;
    }

    // Prepare tensors for arithmetic tests.
    Tensor a(1, 1, 2, 2);
    Tensor b(1, 1, 2, 2);

    a(0, 0, 0, 0) = 1.0;
    a(0, 0, 0, 1) = 2.0;
    a(0, 0, 1, 0) = 3.0;
    a(0, 0, 1, 1) = 4.0;

    b.fill(2.0);

    // Test 5: Addition.
    Tensor sum = a.add(b);

    if (sum(0, 0, 0, 0) != 3.0 ||
        sum(0, 0, 1, 1) != 6.0)
    {
        cerr << "FAILED: Addition\n";
        failures++;
    }

    // Test 6: Subtraction.
    Tensor difference = a.subtract(b);

    if (difference(0, 0, 0, 0) != -1.0 ||
        difference(0, 0, 1, 1) != 2.0)
    {
        cerr << "FAILED: Subtraction\n";
        failures++;
    }

    // Test 7: Element-wise multiplication.
    Tensor product = a.multiply(b);

    if (product(0, 0, 0, 0) != 2.0 ||
        product(0, 0, 1, 1) != 8.0)
    {
        cerr << "FAILED: Element-wise multiplication\n";
        failures++;
    }

    // Test 8: Scalar multiplication.
    Tensor scaled = a.scale(3.0);

    if (scaled(0, 0, 0, 0) != 3.0 ||
        scaled(0, 0, 1, 1) != 12.0)
    {
        cerr << "FAILED: Scalar multiplication\n";
        failures++;
    }

    // Test 9: Reshape while preserving the number of elements.
    a.reshape(1, 2, 1, 2);

    if (a.shape != vector<size_t>{1, 2, 1, 2} ||
        a.size() != 4)
    {
        cerr << "FAILED: Reshape\n";
        failures++;
    }

    // Test 10: Reject an invalid index.
    bool threw = false;

    try
    {
        a(1, 0, 0, 0);
    }
    catch (const out_of_range&)
    {
        threw = true;
    }

    if (!threw)
    {
        cerr << "FAILED: Out-of-range index accepted\n";
        failures++;
    }

    // Test 11: Reject arithmetic with incompatible shapes.
    threw = false;

    try
    {
        Tensor wrong(1, 1, 1, 4);
        a.add(wrong);
    }
    catch (const invalid_argument&)
    {
        threw = true;
    }

    if (!threw)
    {
        cerr << "FAILED: Incompatible shapes accepted\n";
        failures++;
    }

    // Test 12: Reject reshape that changes the element count.
    threw = false;

    try
    {
        a.reshape(1, 1, 3, 2);
    }
    catch (const invalid_argument&)
    {
        threw = true;
    }

    if (!threw)
    {
        cerr << "FAILED: Invalid reshape accepted\n";
        failures++;
    }

    if (failures == 0)
    {
        cout << "All 12 Tensor tests passed." << endl;
        return 0;
    }

    cerr << failures << " Tensor test(s) failed." << endl;
    return 1;
}
