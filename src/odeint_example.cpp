#include <iostream>
#include <vector>
#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>
#include "utils.cuh"

using namespace std;
using namespace std::chrono;
using namespace boost::numeric::odeint;

// typedef float* state_type;
// void rhs(state_type &y, state_type &dydt, const double t);
// float* tau_forward(float (*L)[114], float* tau, float t, float kappa, float rho, float gamma);

float L[114][114];

typedef boost::array<float, 228> state_type;
void tau_forward(float (*L)[114], const float t, const float kappa, const float rho, const float gamma, const state_type& tau, state_type& tau_out) {
    const size_t DIM = 114;
    // TODO: double check if they're inclusive
    boost::array<float, DIM> tau_a;
    boost::array<float, DIM> tau_n;
    for (size_t i = 0; i < DIM; ++i) {
        tau_a[i] = tau[i];
        tau_n[i] = tau[DIM+i];
    }
    boost::array<float, 114> matvec_out;
    // for (int i=0;i<DIM;i++){
    //     for (int j=0; j<DIM; j++){
    //         if (j == 0) matvec_out[i] = L[i][j] * tau_a[j];
    //         else matvec_out[i] += L[i][j] * tau_a[j];
    //     }
    // }
    matvec_func(L, tau_a, matvec_out);
    boost::array<float, DIM> rhs_a;
    boost::array<float, DIM> rhs_n;
    for (int i=0; i<DIM; i++){
        rhs_a[i] = -kappa*(matvec_out[i]) + rho*tau_a[i]*tau_n[i] - gamma*tau_a[i];
        rhs_n[i] = -rho*tau_a[i]*tau_n[i];
    }
    for (size_t i = 0; i < DIM; ++i) {
        tau_out[i] = rhs_a[i];
        tau_out[DIM+i] = rhs_n[i];
    }
}

// typedef std::vector<float> state_type;
// void tau_forward(float (*L)[114], const float t, const float kappa, const float rho, const float gamma, const state_type& tau, state_type& tau_out) {
//     const size_t DIM = 114;

//     std::vector<float> tau_a(tau.begin(), tau.begin()+DIM);
//     std::vector<float> tau_n(tau.begin()+DIM, tau.end());
//     float* matvec_out = new float[114];
//     for (int i=0;i<114;i++){
//         for (int j=0; j<114; j++){
//             if (j == 0) matvec_out[i] = L[i][j] * tau_a[j];
//             else matvec_out[i] += L[i][j] * tau_a[j];
//         }
//     }
    
//     std::vector<float> rhs_a(114);
//     std::vector<float> rhs_n(114);
//     for (int i=0; i<114; i++){
//         rhs_a[i] = - kappa*(matvec_out[i]) + rho*tau_a[i]*tau_n[i] - gamma*tau_a[i];
//         rhs_n[i] = - rho*tau_a[i]*tau_n[i];
//     }
//     tau_out.clear();
//     tau_out.insert(tau_out.end(), rhs_a.begin(), rhs_a.end());
//     tau_out.insert(tau_out.end(), rhs_n.begin(), rhs_n.end());
// }


void rhs(const state_type &y, state_type &dydt, const float t)
{
    tau_forward(L, t, 4, 5, 1, y, dydt);
}

//  void dummy_observer( const state_type &x, const double t )
//  {
// //    std::cout << t << setw(15) << x[0] << setw(15) << x[1] << std::endl;
//  }

int main()
{
    // Define initial conditions
    // state_type c0(228);
    state_type c0;
    for (int i=0; i<228; i++){
        if (i<114){
            if ((i==66) | (i==68)) {
                c0[i] = 1;
            }
            else{
                c0[i] = 0;
            }
        }
        else{
            c0[i] = 1 - c0[i - 114];
        }
    }

    const double t_start = 0.0;
    const double t_end = 1.0;
    const double dt = 0.01;

    for (int i=0; i<114; i++){
        for (int j=0; j<114; j++){
            if (i==j){
                L[i][j] = 1;
            }
            else {
                L[i][j] = -0.00885;
            }
        }
    }

    // Define stepper and integrate the ODE
    runge_kutta4<state_type> stepper;

    auto start = high_resolution_clock::now();
    integrate_const(stepper, rhs, c0, t_start, t_end, dt);
    auto stop = high_resolution_clock::now();

    // for (int i=0; i<228; i++){
    //     cout << c0[i] << ' ';
    // }
    // cout << endl;
    auto duration = duration_cast<microseconds>(stop - start);
    cout << duration.count() << endl;


    return 0;
}
