#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N],id[N],test[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    id[i] = i;
    fx[i] = fy[i] = 0;
  }
  
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  __m256 idvec = _mm256_load_ps(id);

  for(int i=0; i<N; i++) {

    __m256 ivec = _mm256_set1_ps(i);
    __m256 mask = _mm256_cmp_ps(idvec, ivec, _CMP_NEQ_OQ);
    __m256 xveci = _mm256_set1_ps(x[i]);
    __m256 yveci = _mm256_set1_ps(y[i]);

    __m256 rxvec = _mm256_sub_ps(xveci, xvec);
    __m256 ryvec = _mm256_sub_ps(yveci, yvec);

    __m256 revrvec = _mm256_div_ps(_mm256_set1_ps(1.0), _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(rxvec,rxvec),_mm256_mul_ps(ryvec,ryvec))));
    __m256 mrvec = _mm256_mul_ps(mvec,_mm256_mul_ps(revrvec,_mm256_mul_ps(revrvec,revrvec)));

    __m256 fxvec = _mm256_blendv_ps(_mm256_setzero_ps(),_mm256_mul_ps(rxvec,mrvec),mask); 
    __m256 fyvec = _mm256_blendv_ps(_mm256_setzero_ps(),_mm256_mul_ps(ryvec,mrvec),mask); 


    __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(fxvec, 1), _mm256_castps256_ps128(fxvec));
    __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x1));
    fx[i] =  -(_mm_cvtss_f32(x32));


    __m128 y128 = _mm_add_ps(_mm256_extractf128_ps(fyvec, 1), _mm256_castps256_ps128(fyvec));
    __m128 y64 = _mm_add_ps(y128, _mm_movehl_ps(y128, y128));
    __m128 y32 = _mm_add_ss(y64, _mm_shuffle_ps(y64, y64, 0x1));
    fy[i] = -(_mm_cvtss_f32(y32));

    printf("%d %g %g\n",i,fx[i],fy[i]);

/*
    for(int j=0; j<N; j++) {
      if(i != j) {
        float rx = x[i] - x[j];
        float ry = y[i] - y[j];
        float r = std::sqrt(rx * rx + ry * ry);
        fx[i] -= rx * m[j] / (r * r * r);
        fy[i] -= ry * m[j] / (r * r * r);
      }
    }
    printf("%d %g %g\n",i,fx[i],fy[i]);
*/
  }

}
