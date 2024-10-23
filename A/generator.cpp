#include <bits/stdc++.h>

using namespace std;

int main(){
   int n, k;

   cin >> n >> k;
   cout << n << ' ' << k << endl;
   srand(time(NULL));
   for (int i = 0; i < n; i++){
      cout << (rand() % 50) + 1 << ' ';
   }
   cout << endl;
   for (int i = 0; i < k; i++){
      cout << (rand() % 50) + 1 << ' ';
   }
   cout << endl;

   return 0;
}
