#include <iostream>
#include <vector>
#include <boost/numeric/odeint.hpp>
#include <thrust/device_vector.h>

// Define the matrix-vector multiplication kernel
__global__ void matrix_vector_multiply_kernel(double* L, double* ca, double* L_ca, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += L[idx * N + j] * ca[j];
        }
        L_ca[idx] = sum;
    }
}

// Define the ODE system
struct ode_system {
    const double* L_ptr;
    double kappa, rho, gamma;

    __host__ __device__
    void operator()(const thrust::device_vector<double>& x, thrust::device_vector<double>& dxdt, const double t) const {
        int N = x.size() / 2;
        double* ca = thrust::raw_pointer_cast(&x[0]);
        double* cn = thrust::raw_pointer_cast(&x[N]);

        thrust::device_vector<double> L_ca(N);
        matrix_vector_multiply_kernel<<<(N+127)/128, 128>>>(L_ptr, ca, thrust::raw_pointer_cast(L_ca.data()), N);

        for (int i = 0; i < N; ++i) {
            dxdt[i] = - kappa * L_ca[i] + rho * ca[i] * cn[i] - gamma * ca[i];
            dxdt[i+N] = - rho * ca[i] * cn[i];
        }
    }
};

int main() {
    // Define the ODE parameters
    int N = 114;
    double kappa = 4, rho = 5, gamma = 1;
    thrust::device_vector<double> L(N*N);
    thrust::device_vector<double> ca(N), cn(N);

    // Initialize the parameters
    for (int i=0; i<114; i++){
        for (int j=0; j<114; j++){
            if (i==j){
                L[i*114+j] = 1;
            }
            else {
                L[i*114+j] = -0.00885;
            }
        }
    }
    thrust::fill(ca.begin(), ca.end(), 0);
    ca[66] = 1;
    ca[68] = 1;
    thrust::fill(cn.begin(), cn.end(), 1);
    cn[66] = 0;
    cn[68] = 0;

    // Define the ODE solver
    double t0 = 0.0, tf = 1.0;
    double dt = 0.01;
    thrust::device_vector<double> x(N*2);
    thrust::copy(ca.begin(), ca.end(), x.begin());
    thrust::copy(cn.begin(), cn.end(), x.begin()+N);

    ode_system system = {thrust::raw_pointer_cast(L.data()), kappa, rho, gamma};
    boost::numeric::odeint::runge_kutta4_classic< thrust::device_vector<double> > stepper;
    boost::numeric::odeint::integrate_const(stepper, system, x, t0, tf, dt);

    // Print the results
    thrust::copy(x.begin(), x.end(), ca.begin());
    thrust::copy(x.begin()+N, x.end(), cn.begin());

    std::cout << "ca: ";
    for (int i = 0; i < N; ++i) {
        std::cout << ca[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "cn: ";
    for (int i = 0; i < N; ++i) {
        std::cout << cn[i] << " ";
    }
    std::cout << std::endl;

    return 
}