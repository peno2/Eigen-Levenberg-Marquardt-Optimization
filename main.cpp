#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cassert>
#include <Eigen/Eigen>

#include <unsupported/Eigen/NonLinearOptimization>

double warp2(const Eigen::VectorXf &x) { return  fabs(x(0) - 16) + fabs(x(1) - 8); }
double warp3(const Eigen::VectorXf &x) { return  fabs(x(0) - 15) + fabs(x(1) - 7) + fabs(x(2) + 9); }

struct LMFunctor
{
  LMFunctor(int mIn, int nIn): m(mIn), n(nIn) {}
	// 'm' pairs of (x, f(x))
	//Eigen::MatrixXf measuredValues;

	// Compute 'm' errors, one for each data point, for the given parameter values in 'x'
	int operator()(const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const
	{
	  //std::cout << "fvec.size(): " << fvec.size() << std::endl;
	  
		// 'x' has dimensions n x 1
		// It contains the current estimates for the parameters.

		// 'fvec' has dimensions m x 1
		// It will contain the error for each data point.

		fvec(0) = warp2(x);
		//fvec(0) = warp3(x);
		//fvec(0) = fabs(aParam - 15);
		//fvec(1) = fabs(bParam - 7);
		//fvec(2) = fabs(cParam + 9) + 2;
		return 0;
	}

	// Compute the jacobian of the errors
	int df(const Eigen::VectorXf &x, Eigen::MatrixXf &fjac) const
	{
	  assert(x.size() == n);
		// 'x' has dimensions n x 1
		// It contains the current estimates for the parameters.

		// 'fjac' has dimensions m x n
		// It will contain the jacobian of the errors, calculated numerically in this case.

		float epsilon;
		epsilon = 1e-5f;

		for (int i = 0; i < x.size(); ++i) {
			Eigen::VectorXf xPlus(x);
			xPlus(i) += epsilon;
			Eigen::VectorXf xMinus(x);
			xMinus(i) -= epsilon;

			Eigen::VectorXf fvecPlus(values());
			operator()(xPlus, fvecPlus);

			Eigen::VectorXf fvecMinus(values());
			operator()(xMinus, fvecMinus);

			Eigen::VectorXf fvecDiff(values());
			fvecDiff = (fvecPlus - fvecMinus) / (2.0f * epsilon);

			fjac.block(0, i, values(), 1) = fvecDiff;
		}

		return 0;
	}

	// Number of data points, i.e. values.
	int m;

	// Returns 'm', the number of values.
	int values() const { return m; }

	// The number of parameters, i.e. inputs.
	int n;

	// Returns 'n', the number of inputs.
	int inputs() const { return n; }

};


int main(int argc, char *argv[])
{
	//
	// Goal
	//
	// Given a non-linear equation: f(x) = a(x^2) + b(x) + c
	// and 'm' data points (x1, f(x1)), (x2, f(x2)), ..., (xm, f(xm))
	// our goal is to estimate 'n' parameters (3 in this case: a, b, c)
	// using LM optimization.
	//

	//
	// Read values from file.
	// Each row has two numbers, for example: 5.50 223.70
	// The first number is the input value (5.50) i.e. the value of 'x'.
	// The second number is the observed output value (223.70),
	// i.e. the measured value of 'f(x)'.
	//

	// 'm' is the number of data points.
	const int m = 3;//x_values.size();


	// Move the data into an Eigen Matrix.
	// The first column has the input values, x. The second column is the f(x) values.
#if 0
	Eigen::MatrixXf measuredValues(m, 2);
	for (int i = 0; i < m; ++i) {
	  measuredValues(i, 0) = 0;//x_values[i];
	  measuredValues(i, 1) = 0;//y_values[i];
	}
#endif
	// 'n' is the number of parameters in the function.
	// f(x) = a(x^2) + b(x) + c has 3 parameters: a, b, c
	const int n = 3;

	// 'x' is vector of length 'n' containing the initial values for the parameters.
	// The parameters 'x' are also referred to as the 'inputs' in the context of LM optimization.
	// The LM optimization inputs should not be confused with the x input values.
	Eigen::VectorXf x(n);
	//x(0) = 0.0;             // initial value for 'a'
	//x(1) = 0.0;             // initial value for 'b'
	//x(2) = 0.0;             // initial value for 'c'

	//
	// Run the LM optimization
	// Create a LevenbergMarquardt object and pass it the functor.
	//

	LMFunctor functor(m, x.size());
	std::cout << "m: " << m << ", n: " << n << std::endl;

	Eigen::LevenbergMarquardt<LMFunctor, float> lm(functor);
	const int status = lm.minimize(x);
	std::cout << "LM optimization status: " << status << std::endl;

	//
	// Results
	// The 'x' vector contains the results of the optimization.
	//
	std::cout << "Optimization results, x:\n" << x << std::endl;

	//Eigen::VectorXf fvec(1);
	//functor(x, fvec);
	//std::cout << "\terr: " << fvec(0) << std::endl;

	return 0;
}
