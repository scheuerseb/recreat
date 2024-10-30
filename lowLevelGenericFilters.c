#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

// Compile into shared library
// x86_64-w64-mingw32-gcc -shared -fpic lowLevelDiversityFilter.c -o lowLevelDiversityFilter.dll


int sum_filter(
    double * buffer,
    intptr_t filter_size,
    double * return_value,
    void *   user_data
) {
    
    double sum = 0;   
    for(int i=0; i < filter_size; i++) {
        sum = sum + buffer[i];
    }

    *return_value = sum;
    // return 1 to indicate success (CPython convention)
    return 1;
    

}


int div_filter(
    double * buffer,
    intptr_t filter_size,
    double * return_value,
    void *   user_data
) {
    double vals[filter_size];
    
    // c ... class of interest
    double c = *(double *)user_data;
    int div = 0;

    // do we have a kernel that is of interest?
    // i.e., should we ignore this pixel?
    bool hasRelevantClass = false;
    for(int i=0; i < filter_size; i++) {
        if (buffer[i] == c) {
            // this pixel interests us. continue
            hasRelevantClass = true;
            break;
        }        
    }

    if (hasRelevantClass == true) {
        for(int i=0; i < filter_size; i++) {
       
            // Check if this element is included in result
            int j;
            for (j = 0; j < i; j++)
                if (buffer[i] == buffer[j])
                    break;

            // Include this element if not included previously
            if (i == j) {                
                div++;                
            }
            
        }

        *return_value = (double)div;
        // return 1 to indicate success (CPython convention)
        return 1;
    
    } else {
        // simply return 1 as result, since we are not interested in this pixel
        *return_value = (double)1;
        return 1;
    }   
}




int div_filter_ignore_class(
    double * buffer,
    intptr_t filter_size,
    double * return_value,
    void *   user_data
) {
    double vals[filter_size];
    
    // user data holds [0].. land-use class of interest; [1].. ignore class
    int *c = (int *)user_data;

    int div = 0;
   
    // do we have a kernel that is of interest?
    // i.e., should we ignore this pixel?
    bool hasRelevantClass = false;
    for(int i=0; i < filter_size; i++) {
        if (buffer[i] == c[0]) {
            // this pixel interests us. continue
            hasRelevantClass = true;
            break;
        }        
    }

    if (hasRelevantClass == true) {
        for(int i=0; i < filter_size; i++) {
       
            // Check if this element is included in result
            int j;
            for (j = 0; j < i; j++)
                if (buffer[i] == buffer[j])
                    break;

            // Include this element if not included previously
            if (i == j) {
                // if this element is the edge class to be ignored, don't increase div
                if (buffer[i] != c[1]) {
                    div++;
                }
            }
            
        }

        *return_value = (double)div;
        // return 1 to indicate success (CPython convention)
        return 1;

    } else {
        // simply return 1 as result, since we are not interested in this pixel
        *return_value = (double)1;
        return 1;
    }   
}