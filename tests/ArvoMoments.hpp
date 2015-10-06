#pragma once

////////////////////////////////////////////
// James Arvo's Axial Moment
////////////////////////////////////////////

float dot(const float* a, const float* b) {
   return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

void CrossProduct(const float* A, const float* B, float* out) {
   out[0] = -A[2] * B[1] + A[1] * B[2];
   out[1] = A[2] * B[0] - A[0] * B[2];
   out[2] = -A[1] * B[0] + A[0] * B[1];
}

float ArvoCosSumIntegral( float x, float y, float c, int m, int n ) {
   int i = n % 2;
   float F = (i == 0) ? (y - x) : (sin(y) - sin(x));
   float S = 0.0f;
   while( i <= n ) {
      if( i >= m ) {
         S += pow(c,i) * F;
      }
      float T = pow(cos(y),i+1.0f) * sin(y) - pow(cos(x),i+1.0f) * sin(x);
      F = (T + (i+1.0f)*F)/(i+2.0f);
      i += 2;
   }

   return S;
}

float ArvoLineIntegral(const float* A, const float* B, const float* w, int m, int n ) {
   const float epsilon = 1e-7;
   if( (n < 0) || ( (fabs(dot(w,A)) < epsilon) && (fabs(dot(w,B)) < epsilon) ) ) {
      return 0;
   }

   float lenA = sqrt(dot(A,A));
   float lenB = sqrt(dot(B,B));
   float s[3] = { A[0] / lenA, A[1] / lenA, A[2] / lenA };
   float M[3][3] = { {1.0f - s[0]*s[0], -s[0]*s[1], -s[0]*s[2]}, {-s[0]*s[1], 1.0f - s[1]*s[1], -s[1]*s[2]}, {-s[0]*s[2], -s[1]*s[2], 1.0f - s[2]*s[2]} };
   float t[3] = { M[0][0] * B[0] + M[0][1] * B[1] + M[0][2] * B[2], M[1][0] * B[0] + M[1][1] * B[1] + M[1][2] * B[2], M[2][0] * B[0] + M[2][1] * B[1] + M[2][2] * B[2] } ;
   float len_t = sqrt(dot(t,t));
   t[0] /= len_t;
   t[1] /= len_t;
   t[2] /= len_t;
   float a = dot(w,s);
   float b = dot(w,t);
   float c = sqrt(a*a + b*b);
   //float l = acos( dot(A,B) / (lenA * lenB));
   float acosVal = dot(A,B) / (lenA * lenB);
   float l;
   float eps = 1e-6;
   if(std::abs(acosVal-(-1)) < eps)
      l = M_PI;
   else if(std::abs(acosVal-1) < eps)
      l = 0;
   else
      l = acos( dot(A,B) / (lenA * lenB));
   //printf("%f %f\n", dot(A, B), l);
   float signb = (b < 0) ? -1 : 1;
   float phi = signb * acos(a/c);
   return ArvoCosSumIntegral(-phi, l - phi, c, m, n);
}

float ArvoBoundaryIntegral(const float* tri, const float w[3], const float v[3], int m, int n ) {
   float b = 0;

   // For each edge (unrolled):
   // E1
   const float* A = &tri[3];
   const float* B = &tri[0];
   float n1[3];
   CrossProduct(A,B,n1);
   float n1len = sqrt(dot(n1,n1));
   n1[0] /= n1len;
   n1[1] /= n1len;
   n1[2] /= n1len;
   b += dot(n1,v) * ArvoLineIntegral(A, B, w, m, n);

   // E2
   A = &tri[6];
   B = &tri[3];
   float n2[3];
   CrossProduct(A,B,n2);
   float n2len = sqrt(dot(n2,n2));
   n2[0] /= n2len;
   n2[1] /= n2len;
   n2[2] /= n2len;
   b += dot(n2,v) * ArvoLineIntegral(A, B, w, m, n);

   // E3
   A = &tri[0];
   B = &tri[6];
   float n3[3];
   CrossProduct(A,B,n3);
   float n3len = sqrt(dot(n3,n3));
   n3[0] /= n3len;
   n3[1] /= n3len;
   n3[2] /= n3len;
   b += dot(n3,v) * ArvoLineIntegral(A, B, w, m, n);

   return b;
}

float TriangleSolidAngle(const float *v0, const float *v1, const float *v2 ) {
   //float V0[3] = { v0[0] / v0len, v0[1] / v0len, v0[2] / v0len };
   //float V1[3] = { v1[0] / v1len, v1[1] / v1len, v1[2] / v1len };
   //float V2[3] = { v2[0] / v2len, v2[1] / v2len, v2[2] / v2len };
   float V0[3] = { v0[0], v0[1], v0[2] };
   float V1[3] = { v1[0], v1[1], v1[2] };
   float V2[3] = { v2[0], v2[1], v2[2] };

   float temp[3];
   CrossProduct(V1,V2,temp);
   float det = std::abs(dot(temp,V0));
   float al = sqrt(dot(V0,V0));
   float bl = sqrt(dot(V1,V1));
   float cl = sqrt(dot(V2,V2));
   float div = al*bl*cl + dot(V0,V1)*cl + dot(V0,V2)*bl + dot(V1,V2)*al;
   float at = atan2(det, div);
   if(at < 0) at += M_PI; // If det>0 && div<0 atan2 returns < 0, so add pi.
   float omega = 2.0f * at;
   return omega;
}

float ArvoAxialMoment(const float* tri, const float w[3], int n) {
   const float epsilon = 1e-10;
   float a = -ArvoBoundaryIntegral(tri,w,w,0,n-1);
   if( n % 2 == 0   ) { a += TriangleSolidAngle(&tri[0], &tri[3], &tri[6]); } //|| fabs(a) < epsilon

   return a / (n+1);
}