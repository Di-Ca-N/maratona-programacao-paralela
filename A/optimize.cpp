#include<bits/stdc++.h>
using namespace std;

#define MAX 5012

int n, k;
int file_size[MAX];
int memo[MAX][60][60][60];

int dp(int i, int capacity1, int capacity2, int capacity3) {
    int r1, r2, r3, r4;
    r1 = r2 = r3 = r4 = 0;

    if (i == n) return 0;

    if (memo[i][capacity1][capacity2][capacity3] != -1)
        return memo[i][capacity1][capacity2][capacity3];

    #pragma omp task shared(r1)
    r1 = dp(i + 1, capacity1, capacity2, capacity3);

    if (file_size[i] <= capacity1){
        #pragma omp task shared(r2)
        r2 = dp(i + 1, capacity1 - file_size[i], capacity2, capacity3) + file_size[i];
    }
    if (file_size[i] <= capacity2) {
        #pragma omp task shared(r3)
        r3 = dp(i + 1, capacity1, capacity2 - file_size[i], capacity3) + file_size[i];    
    }
    if (file_size[i] <= capacity3) {
        #pragma omp task shared(r4)
        r4 = dp(i + 1, capacity1, capacity2, capacity3 - file_size[i]) + file_size[i];
    }
    #pragma omp taskwait

    int result = max(r1, max(r2, max(r3, r4)));

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

    int result;
    #pragma omp parallel
    {
        #pragma omp single
        {
            result = dp(0, capacity[0], capacity[1], capacity[2]);
        }
    }
    printf("%d\n", result);
    return 0;
}
