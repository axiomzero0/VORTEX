// =============================================================================
// tests/lang/lang_cases.hpp — language-level test cases (Rule 36 differential
// corpus). Each case: Python-subset source + expected stdout.
// =============================================================================

#pragma once

#include <cstddef>

struct LangCase {
    const char* name;
    const char* src;
    const char* expect;
};

inline constexpr LangCase kLangCases[] = {
    {"arith", "print(1 + 2 * 3 - 4 / 2)\nprint(2 ** 10)\nprint(7 // 2)\nprint(7 % 3)\nprint(-5 // 2)\nprint(-5 % 2)\n", "5.0\n1024\n3\n1\n-3\n1\n"},
    {"bigint", "x = 2 ** 80\nprint(x)\nprint(x + 1)\nprint(2 ** 64 * 2 ** 64)\n", "1208925819614629174706176\n1208925819614629174706177\n340282366920938463463374607431768211456\n"},
    {"strings", "s = \"hello\"\nprint(s + \" world\")\nprint(s * 2)\nprint(len(s))\nprint(s[1])\nprint(s[0:3])\nprint(\"ell\" in s)\nprint(\"z\" in s)\n", "hello world\nhellohello\n5\ne\nhel\nTrue\nFalse\n"},
    {"lists", "a = [1, 2, 3]\na.append(4)\nprint(a)\nprint(a[1])\na[0] = 9\nprint(a)\nprint(len(a))\nprint(sum(a))\nprint(max(a), min(a))\nb = [x * 2 for x in a if x > 1]\nprint(b)\n", "[1, 2, 3, 4]\n2\n[9, 2, 3, 4]\n4\n18\n9 2\n[18, 4, 6, 8]\n"},
    {"while_loop", "i = 0\ns = 0\nwhile i < 5:\n    s = s + i\n    i = i + 1\nprint(s)\n", "10\n"},
    {"for_range", "t = 0\nfor i in range(10):\n    t = t + i\nprint(t)\nfor j in range(2, 8, 2):\n    print(j)\n", "45\n2\n4\n6\n"},
    {"for_list", "xs = [10, 20, 30]\nt = 0\nfor x in xs:\n    t = t + x\nprint(t)\n", "60\n"},
    {"break_continue", "s = 0\nfor i in range(10):\n    if i == 3:\n        continue\n    if i == 7:\n        break\n    s = s + i\nprint(s)\n", "18\n"},
    {"dict", "d = {\"a\": 1, \"b\": 2}\nprint(d[\"a\"])\nd[\"c\"] = 3\nprint(d[\"c\"])\nprint(\"a\" in d)\nprint(len(d))\nfor k in d:\n    print(k)\n", "1\n3\nTrue\n3\na\nb\nc\n"},
    {"functions", "def add(a, b=10):\n    return a + b\nprint(add(5))\nprint(add(5, 1))\ndef var(*args):\n    return sum(args)\nprint(var(1, 2, 3))\n", "15\n6\n6\n"},
    {"recursion", "def fact(n):\n    if n <= 1:\n        return 1\n    return n * fact(n - 1)\nprint(fact(10))\n", "3628800\n"},
    {"closures", "def counter():\n    c = 0\n    def inc():\n        nonlocal c\n        c = c + 1\n        return c\n    return inc\nc1 = counter()\nprint(c1())\nprint(c1())\nprint(c1())\n", "1\n2\n3\n"},
    {"closures_readonly", "def outer():\n    c = 100\n    def get():\n        return c\n    return get\nf = outer()\nprint(f())\n", "100\n"},
    {"classes", "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def mag2(self):\n        return self.x * self.x + self.y * self.y\n\nclass Point3(Point):\n    def __init__(self, x, y, z):\n        Point.__init__(self, x, y)\n        self.z = z\n    def mag2(self):\n        return Point.mag2(self) + self.z * self.z\n\np = Point(3, 4)\nprint(p.mag2())\nq = Point3(1, 2, 3)\nprint(q.mag2())\nprint(q.x)\n", "25\n14\n1\n"},
    {"try_except", "def f(x):\n    try:\n        if x < 0:\n            raise ValueError(\"neg\")\n        return x * 2\n    except ValueError as e:\n        return -1\n    except Exception:\n        return -2\nprint(f(5))\nprint(f(-1))\n", "10\n-1\n"},
    {"try_finally", "def f():\n    try:\n        return 1\n    finally:\n        pass\nprint(f())\n", "1\n"},
    {"zero_div", "def f():\n    try:\n        return 1 / 0\n    except ZeroDivisionError:\n        return 99\nprint(f())\n", "99\n"},
    {"generators", "def gen(n):\n    for i in range(n):\n        yield i * i\nt = 0\nfor v in gen(5):\n    t = t + v\nprint(t)\nprint(sum(x for x in gen(4)))\n", "30\n14\n"},
    {"comprehension", "m = [[1, 2], [3, 4]]\nflat = [x for row in m for x in row]\nprint(flat)\n", "[1, 2, 3, 4]\n"},
    {"tuple_unpack", "a, b = 1, 2\nprint(a + b)\nx, y = [10, 20]\nprint(x + y)\n", "3\n30\n"},
    {"bool_shortcircuit", "def t():\n    print(\"t\")\n    return True\ndef f():\n    print(\"f\")\n    return False\nr = f() and t()\nprint(r)\nr = t() or f()\nprint(r)\nprint(1 < 2 < 3)\nprint(3 > 2 > 1 > 0)\nprint(1 < 2 > 5)\n", "f\nFalse\nt\nTrue\nTrue\nTrue\nFalse\n"},
    {"ternary", "x = 5\nprint(\"big\" if x > 3 else \"small\")\n", "big\n"},
    {"lambda", "f = lambda x: x * 3\nprint(f(4))\ng = lambda a, b: a + b\nprint(g(2, 3))\n", "12\n5\n"},
    {"str_methods", "s = \"Hello World\"\nprint(s.upper())\nprint(s.lower())\nprint(s.split(\" \"))\nprint(s.startswith(\"Hello\"))\nprint(len(s))\n", "HELLO WORLD\nhello world\n['Hello', 'World']\nTrue\n11\n"},
    {"slices", "s = \"hello world\"\nprint(s[0:5])\nprint(s[6:])\nprint(s[:5])\nprint(s[-5:])\nprint(s[::2])\nprint(s[::-1])\nxs = [0, 1, 2, 3, 4, 5]\nprint(xs[1:4])\nprint(xs[::-1])\nprint(xs[::2])\n", "hello\nworld\nhello\nworld\nhlowrd\ndlrow olleh\n[1, 2, 3]\n[5, 4, 3, 2, 1, 0]\n[0, 2, 4]\n"},
    {"import_math", "import math\nprint(math.sqrt(16))\nprint(math.floor(3.7))\nprint(math.pi > 3.14 and math.pi < 3.15)\n", "4.0\n3\nTrue\n"},
    {"import_from", "from math import sqrt\nprint(sqrt(25))\n", "5.0\n"},
    {"nested_data", "def build():\n    result = []\n    for i in range(3):\n        row = []\n        for j in range(3):\n            row.append(i * 3 + j)\n        result.append(row)\n    return result\nm = build()\nprint(m)\nprint(m[1][2])\n", "[[0, 1, 2], [3, 4, 5], [6, 7, 8]]\n5\n"},
    {"enumerate_zip", "xs = [10, 20, 30]\nfor i, v in enumerate(xs):\n    print(i, v)\nfor a, b in zip(xs, [1, 2]):\n    print(a + b)\n", "0 10\n1 20\n2 30\n11\n22\n"},
    {"map_filter", "xs = [1, 2, 3, 4, 5]\nys = [x * x for x in xs if x % 2 == 1]\nprint(ys)\nprint(sorted([3, 1, 2]))\n", "[1, 9, 25]\n[1, 2, 3]\n"},
    {"global_fn", "def helper():\n    return 42\n\ndef use():\n    return helper()\nprint(use())\n", "42\n"},
    {"assert_stmt", "x = 1\nassert x == 1\ntry:\n    assert x == 2\nexcept AssertionError:\n    print(\"caught\")\n", "caught\n"},
};

inline constexpr std::size_t kLangCaseCount = sizeof(kLangCases) / sizeof(kLangCases[0]);
