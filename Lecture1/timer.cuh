#ifndef TIMER_CUH
#define TIMER_CUH
#include <math.h>
#include <sys/time.h>

double measureTime(void)
{
    static bool active = false;
    static time_t s;
    static suseconds_t us;
    double ms;
    struct timeval tv;
    struct timezone tz;

    if(active){
        active = false;
        gettimeofday(&tv,&tz);
        ms = 1.0e+3 * (tv.tv_sec - s) + 1.0e-3 * (tv.tv_usec - us);
    }
    else{
        active = true;
        ms = 0.0;
        gettimeofday(&tv,&tz);
        s = tv.tv_sec;
        us = tv.tv_usec;
    }
    return ms;
}

#endif
