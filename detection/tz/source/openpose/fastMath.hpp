#ifndef ALG_OPENPOSE_MX_FASTMATH_A
#define ALG_OPENPOSE_MX_FASTMATH_A

// Round functions
// Signed
template<typename T>
inline char charRound(const T a)
{
    return char(a+0.5f);
}

template<typename T>
inline signed char sCharRound(const T a)
{
    return (signed char)(a+0.5f);
}

template<typename T>
inline int intRound(const T a)
{
    return int(a+0.5f);
}

template<typename T>
inline long longRound(const T a)
{
    return long(a+0.5f);
}

template<typename T>
inline long long longLongRound(const T a)
{
    return (long long)(a+0.5f);
}

// Unsigned
template<typename T>
inline unsigned char uCharRound(const T a)
{
    return (unsigned char)(a+0.5f);
}

template<typename T>
inline unsigned int uIntRound(const T a)
{
    return (unsigned int)(a+0.5f);
}

template<typename T>
inline unsigned long ulongRound(const T a)
{
    return (unsigned long)(a+0.5f);
}

template<typename T>
inline unsigned long long uLongLongRound(const T a)
{
    return (unsigned long long)(a+0.5f);
}

// Max/min functions
template<typename T>
inline T fastMax(const T a, const T b)
{
    return (a > b ? a : b);
}

template<typename T>
inline T fastMin(const T a, const T b)
{
    return (a < b ? a : b);
}

template<class T>
inline T fastTruncate(T value, T min = 0, T max = 1)
{
    return fastMin(max, fastMax(min, value));
}
#endif
