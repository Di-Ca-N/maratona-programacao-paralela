#include<bits/stdc++.h>

using namespace std;

#define MAX 5012
#define THRESH 200

int n, k;
int file_size[MAX];
int memo[MAX][60][60][60];


int dp_serial(int i, int capacity1, int capacity2, int capacity3) {
    int r1, r2, r3, r4;
    r1 = r2 = r3 = r4 = 0;

    if (i == n) return 0;

    int value;
    #pragma omp atomic read
    value = memo[i][capacity1][capacity2][capacity3];

    if (value != -1)
        return value;

    r1 = dp_serial(i + 1, capacity1, capacity2, capacity3);


    if (file_size[i] <= capacity1){
        r2 = dp_serial(i + 1, capacity1 - file_size[i], capacity2, capacity3) + file_size[i];
    }

    if (file_size[i] <= capacity2) {
        r3 = dp_serial(i + 1, capacity1, capacity2 - file_size[i], capacity3) + file_size[i];   
    }

    if (file_size[i] <= capacity3) {
        r4 = dp_serial(i + 1, capacity1, capacity2, capacity3 - file_size[i]) + file_size[i];
    }
    int result = max(r1, max(r2, max(r3, r4)));
    #pragma omp atomic write
    memo[i][capacity1][capacity2][capacity3] = result;
    return result;
}

int dp(int i, int capacity1, int capacity2, int capacity3) {
    int r1, r2, r3, r4;

    r1 = r2 = r3 = r4 = 0;

    if (i == n) return 0;
    int value;
    #pragma omp atomic read
    value = memo[i][capacity1][capacity2][capacity3];

    if (value != -1)
        return value;

    #pragma omp task shared(r1) priority(i+1) if(n - i < THRESH) untied
    if (n - i < THRESH) {
        r1 = dp_serial(i + 1, capacity1, capacity2, capacity3);
    } else {
        r1 = dp(i + 1, capacity1, capacity2, capacity3);
    }
    

    if (file_size[i] <= capacity1){
        #pragma omp task shared(r2) priority(i+1) if(n - i < THRESH) untied
        if (n - i < THRESH) {
            r2 = dp_serial(i + 1, capacity1 - file_size[i], capacity2, capacity3) + file_size[i];
        } else {
            r2 = dp(i + 1, capacity1 - file_size[i], capacity2, capacity3) + file_size[i];
        }
    }

    if (file_size[i] <= capacity2) {
        #pragma omp task shared(r3) priority(i+1) if(n - i < THRESH) untied
        if (n - i < THRESH) {
            r3 = dp_serial(i + 1, capacity1, capacity2 - file_size[i], capacity3) + file_size[i];
        } else {
            r3 = dp(i + 1, capacity1, capacity2 - file_size[i], capacity3) + file_size[i];
        }
    }

    if (file_size[i] <= capacity3) {
        #pragma omp task shared(r4) priority(i+1) if(n - i < THRESH) untied
        if (n - i < THRESH) {
            r4 = dp_serial(i + 1, capacity1, capacity2, capacity3 - file_size[i]) + file_size[i];
        } else {
            r4 = dp(i + 1, capacity1, capacity2, capacity3 - file_size[i]) + file_size[i];
        }
    }
    #pragma omp taskwait

    //printf("%d %d %d %d\n", r1, r2, r3, r4);
    int result = max(r1, max(r2, max(r3, r4)));
    #pragma omp atomic write
    memo[i][capacity1][capacity2][capacity3] = result;
    return result;
}

int main(){
    int i;
    int capacity[3] = {0, 0, 0};
    
    scanf("%d %d", &n, &k);
    
    for(i = 0; i < n; i++) { 
        scanf("%d", &file_size[i]);
    }

    for (i = 0; i < k; i++){
        scanf("%d", &capacity[i]);
    } 
    
    memset(memo, -1, sizeof(memo));
    // #pragma omp parallel for collapse(4) schedule(static, 64)
    // for (int i = 0; i < MAX; i++) {
    //     for (int j = 0; j < 60; j++) {
    //         for (int k = 0; k < 60; k++) {
    //             for (int l = 0; l < 60; l++) {
    //                 memo[i][j][k][l] = -1;
    //             }
    //         }
    //     }
    // }

    int result = 0;
    #pragma omp parallel shared(memo)
    {
        #pragma omp single
        {
            result = dp(0, capacity[0], capacity[1], capacity[2]);
        }
    }
    printf("%d\n", result);

    return 0;
}
