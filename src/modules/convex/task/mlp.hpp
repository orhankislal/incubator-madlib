/* ----------------------------------------------------------------------- *//**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 * @file mlp.hpp
 *
 * This file contains objective function related computation, which is called
 * by classes in algo/, e.g.,  loss, gradient functions
 *
 *//* ----------------------------------------------------------------------- */

#ifndef MADLIB_MODULES_CONVEX_TASK_MLP_HPP_
#define MADLIB_MODULES_CONVEX_TASK_MLP_HPP_

#include <math.h>
#include <dbconnector/dbconnector.hpp>

namespace madlib {

namespace modules {

namespace convex {

// Use Eigen
using namespace madlib::dbal::eigen_integration;

template <class Model, class Tuple>
class MLP {
public:
    typedef Model model_type;
    typedef Tuple tuple_type;
    typedef typename Tuple::independent_variables_type
        independent_variables_type;
    typedef typename Tuple::dependent_variable_type dependent_variable_type;

    static void gradientInPlace(
            model_type                          &model,
            const independent_variables_type    &x,
            const dependent_variable_type       &y,
            const double                        &stepsize);

    static double getLossAndUpdateModel(
            model_type              &model,
            const Matrix            &x_batch,
            const Matrix            &y_true_batch,
            const double            &stepsize,
            const int               &opt_code,
            const double            &gamma,
            std::vector<Matrix>     &sqrs,
            const double            &beta1,
            const double            &beta2,
            std::vector<Matrix>     &vs,
            const int               &t);

    static double getLossAndGradient(
            model_type                    &model,
            const Matrix                        &x,
            const Matrix                        &y,
            std::vector<Matrix>                 &total_gradient_per_layer,
            const double                        &stepsize);

    static double getLoss(
            const ColumnVector                  &y_true,
            const ColumnVector                  &y_estimated,
            const bool                          is_classification);

    static double loss(
            const model_type                    &model,
            const independent_variables_type    &x,
            const dependent_variable_type       &y);

    static ColumnVector predict(
            const model_type                    &model,
            const independent_variables_type    &x,
            const bool                          is_classification_response,
            const bool                          is_dep_var_array_for_classification);

    const static int RELU = 0;
    const static int SIGMOID = 1;
    const static int TANH = 2;
    static double lambda;

private:

    static const int IS_RMSPROP = 1;
    static const int IS_ADAM = 2;

    static double sigmoid(const double &xi) {
        return 1. / (1. + std::exp(-xi));
    }

    static double relu(const double &xi) {
        return xi*(xi>0);
    }

    static double tanh(const double &xi) {
        return std::tanh(xi);
    }

    static double sigmoidDerivative(const double &xi) {
        double value = sigmoid(xi);
        return value * (1. - value);
    }

    static double reluDerivative(const double &xi) {
        return xi>0;
    }

    static double tanhDerivative(const double &xi) {
        double value = tanh(xi);
        return 1-value*value;
    }

    static void feedForward(
            const model_type                    &model,
            const independent_variables_type    &x,
            std::vector<ColumnVector>           &net,
            std::vector<ColumnVector>           &o);

    static void backPropogate(
            const ColumnVector                  &y_true,
            const ColumnVector                  &y_estimated,
            const std::vector<ColumnVector>     &net,
            const model_type                    &model,
            std::vector<ColumnVector>           &delta);
};

template <class Model, class Tuple>
double MLP<Model, Tuple>::lambda = 0;

template <class Model, class Tuple>
double
MLP<Model, Tuple>::getLoss(const ColumnVector &y_true,
                           const ColumnVector &y_estimated,
                           const bool is_classification){
    if(is_classification){
        // cross entropy loss function
        double clip = 1.e-10;
        ColumnVector y_clipped = y_estimated.cwiseMax(clip).cwiseMin(1. - clip);
        return -(y_true.array() * y_clipped.array().log() +
                    (1 - y_true.array()) * (1 - y_clipped.array()).log()
                ).sum();
    }
    else{
        // squared loss
        return 0.5 * (y_estimated - y_true).squaredNorm();
    }
}

/*
template <class Model, class Tuple>
double
MLP<Model, Tuple>::getLossAndUpdateModel(
        model_type              &model,
        const Matrix            &x_batch,
        const Matrix            &y_true_batch,
        const double            &stepsize,
        const int               &opt_code,
        const double            &gamma,
        std::vector<Matrix>     &sqrs,
        const double            &beta1,
        const double            &beta2,
        std::vector<Matrix>     &vs,
        const int               &t) {

    double total_loss = 0.;
    double eps_stable = 1.e-18;

    // initialize gradient vector
    std::vector<Matrix> total_gradient_per_layer(model.num_layers);
    Matrix g, v_bias_corr, sqr_bias_corr;
    for (Index k=0; k < model.num_layers; ++k) {
        total_gradient_per_layer[k] = Matrix::Zero(model.u[k].rows(),
                                                   model.u[k].cols());
    }

    std::vector<ColumnVector> net, o, delta;
    Index num_rows_in_batch = x_batch.rows();
    for (Index i=0; i < num_rows_in_batch; i++){
        // gradient and loss added over the batch
        ColumnVector x = x_batch.row(i);
        ColumnVector y_true = y_true_batch.row(i);

        feedForward(model, x, net, o);
        backPropogate(y_true, o.back(), net, model, delta);

        // compute the gradient
        for (Index k=0; k < model.num_layers; k++){
                total_gradient_per_layer[k] += o[k] * delta[k].transpose();
        }

        // compute the loss
        total_loss += getLoss(y_true, o.back(), model.is_classification);
    }

    for (Index k=0; k < model.num_layers; k++){
        // convert gradient to a gradient update vector
        //  1. normalize to per row update
        //  2. discount by stepsize
        //  3. add regularization
        //  4. make negative for descent
        Matrix regularization = MLP<Model, Tuple>::lambda * model.u[k];
        regularization.row(0).setZero(); // Do not update bias

        g = total_gradient_per_layer[k] / static_cast<double>(num_rows_in_batch) +
            regularization;
        if (opt_code == IS_RMSPROP){

            //for param, sqr in zip(params, sqrs):
                // g = param.grad / batch_size
                // sqr[:] = gamma * sqr + (1. - gamma) * nd.square(g)
                // div = lr * g / nd.sqrt(sqr + eps_stable)
                // param[:] -= div

            sqrs[k] = gamma * sqrs[k] + (1.0 - gamma) * square(g);
            total_gradient_per_layer[k] = (-stepsize * g).array() /
                                          (sqrs[k].array() + eps_stable).sqrt();
        }
        else if (opt_code == IS_ADAM){

            // for param, v, sqr in zip(params, vs, sqrs):
            //     g = param.grad / batch_size

            //     v[:] = beta1 * v + (1. - beta1) * g
            //     sqr[:] = beta2 * sqr + (1. - beta2) * nd.square(g)

            //     v_bias_corr = v / (1. - beta1 ** t)
            //     sqr_bias_corr = sqr / (1. - beta2 ** t)

            //     div = lr * v_bias_corr / (nd.sqrt(sqr_bias_corr) + eps_stable)
            //     param[:] = param - div
            vs[k] = beta1 * vs[k] + (1.0-beta1) * g;
            sqrs[k] = beta2 * sqrs[k] + (1.0 - beta2) * square(g);

            v_bias_corr = vs[k] / (1. - pow(beta1,t));
            sqr_bias_corr = sqrs[k] / (1. - pow(beta2,t));

            total_gradient_per_layer[k] = (-stepsize * v_bias_corr).array() /
                                          (sqr_bias_corr.array().sqrt() + eps_stable);

        }
        else {
            total_gradient_per_layer[k] = -stepsize * g;
        }

        // total_gradient_per_layer is now the update vector
        if (model.momentum > 0 && opt_code == 0){
            model.velocity[k] = model.momentum * model.velocity[k] + total_gradient_per_layer[k];
            if (model.is_nesterov){
                // Below equation ensures that Nesterov updates are half step
                // ahead of regular momentum updates i.e. next step's discounted
                // velocity update is already added in the current step.
                model.u[k] += model.momentum * model.velocity[k] + total_gradient_per_layer[k];
            }
            else{
                model.u[k] += model.velocity[k];
            }
        } else {
            // no momentum
            model.u[k] += total_gradient_per_layer[k];
        }
    }

    return total_loss;
}
*/

/*
# Rosenbrock Function implementation in R
par <- c(-1.2, 1)

fn <- function(x) {
  x1 <- x[1]
  x2 <- x[2]
  100 * (x2 - x1 * x1) ^ 2 + (1 - x1) ^ 2
}
gr <- function(x) {
  x1 <- x[1]
  x2 <- x[2]
  c(
    -400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1),
    200 *      (x2 - x1 * x1))
}
lr <- 0.001
mu <- 0.95
max_iter <- 100

rmsprop <- function(par, fn, gr, lr, gamma, max_iter = 10) {
  fs <- rep(0, max_iter)

  v <- rep(0, length(par))
  sqr <- rep(0, length(par))
  for (i in 1:max_iter) {
    g <- gr(par)/30

    sqr <- gamma * sqr + (1-gamma) * g * g
    div <- lr * g / sqrt(sqr + 0.00000000001)
    par <- par - div

    # store results
    f <- fn(par)
    fs[i] <- f
  }

  list(par = par, f = f, fs = fs)
}

rms_opt <- rmsprop(par, fn, gr, lr, 0.9, max_iter)

*/
template <class Model, class Tuple>
double
MLP<Model, Tuple>::getLossAndUpdateModel(
        model_type              &model,
        const Matrix            &x_batch,
        const Matrix            &y_true_batch,
        const double            &stepsize,
        const int               &opt_code,
        const double            &gamma,
        std::vector<Matrix>     &sqrs,
        const double            &beta1,
        const double            &beta2,
        std::vector<Matrix>     &vs,
        const int               &t) {

    double total_loss = 0.;
    double eps_stable = 1.e-18;

    // initialize gradient vector
    std::vector<Matrix> total_gradient_per_layer(model.num_layers);
    Matrix g, v_bias_corr, sqr_bias_corr;
    for (Index k=0; k < model.num_layers; ++k) {
        total_gradient_per_layer[k] = Matrix::Zero(model.u[k].rows(),
                                                   model.u[k].cols());
    }

    std::vector<ColumnVector> net, o, delta;
    Index num_rows_in_batch = x_batch.rows();

    double foo = 100 *
                 pow((model.u[0](2, 0) - model.u[0](1, 0) * model.u[0](1, 0)),
                     2) + pow((1 - model.u[0](1, 0)), 2);
    elog(INFO, "rosenbrock loss %f", foo);
    total_loss += foo;

    total_gradient_per_layer[0](1, 0) = (-400 * model.u[0](1, 0) *
                                        (model.u[0](2, 0) -
                                         model.u[0](1, 0) * model.u[0](1, 0)) -
                                        2 * (1 - model.u[0](1, 0)));
    total_gradient_per_layer[0](2, 0) = (200 * (model.u[0](2, 0) - model.u[0](1, 0) * model.u[0](1, 0)));



    if (opt_code == IS_RMSPROP){
        g = total_gradient_per_layer[0] / static_cast<double>(num_rows_in_batch);
        sqrs[0] = gamma * sqrs[0] + (1.0 - gamma) * square(g);
        total_gradient_per_layer[0] = (-stepsize * g).array() /
                                      (sqrs[0].array() + eps_stable).sqrt();
    }else {
        total_gradient_per_layer[0] = -0.0001 * total_gradient_per_layer[0];
    }


    model.u[0] += total_gradient_per_layer[0];

    return total_loss;
}




template <class Model, class Tuple>
void
MLP<Model, Tuple>::gradientInPlace(
        model_type                          &model,
        const independent_variables_type    &x,
        const dependent_variable_type       &y_true,
        const double                        &stepsize)
{
    std::vector<ColumnVector> net, o, delta;

    feedForward(model, x, net, o);
    backPropogate(y_true, o.back(), net, model, delta);

    for (Index k=0; k < model.num_layers; k++){
        Matrix regularization = MLP<Model, Tuple>::lambda*model.u[k];
        regularization.row(0).setZero(); // Do not update bias

        if (model.momentum > 0){
            Matrix gradient = -stepsize * (o[k] * delta[k].transpose() + regularization);
            model.velocity[k] = model.momentum * model.velocity[k] + gradient;
            if (model.is_nesterov)
                model.u[k] += model.momentum * model.velocity[k] + gradient;
            else
                model.u[k] += model.velocity[k];
        }
        else {
            // Updating model inline as a special case to avoid a copy of the
            // gradient matrix to velocity.
            model.u[k] -= stepsize * (o[k] * delta[k].transpose() + regularization);
        }
    }
}

template <class Model, class Tuple>
double
MLP<Model, Tuple>::loss(
        const model_type                    &model,
        const independent_variables_type    &x,
        const dependent_variable_type       &y_true) {

    // Here we compute the loss. In the case of regression we use sum of square errors
    // In the case of classification the loss term is cross entropy.
    std::vector<ColumnVector> net, o;
    feedForward(model, x, net, o);
    ColumnVector y_estimated = o.back();

    return getLoss(y_true, y_estimated, model.is_classification);
}

template <class Model, class Tuple>
ColumnVector
MLP<Model, Tuple>::predict(
        const model_type                    &model,
        const independent_variables_type    &x,
        const bool                          is_classification_response,
        const bool                          is_dep_var_array_for_classification) {
    std::vector<ColumnVector> net, o;

    feedForward(model, x, net, o);
    ColumnVector output = o.back();

    if(is_classification_response){
        int max_idx;
        output.maxCoeff(&max_idx);
        if(is_dep_var_array_for_classification) {
            // Return the entire array, but with 1 for the class level with
            // largest probability and 0s for the rest.
            output.setZero();
            output[max_idx] = 1;
        } else {
            // Return a length 1 array with the predicted index
            output.resize(1);
            output[0] = (double) max_idx;
        }
    }
    return output;
}


template <class Model, class Tuple>
void
MLP<Model, Tuple>::feedForward(
        const model_type                    &model,
        const independent_variables_type    &x,
        std::vector<ColumnVector>           &net,
        std::vector<ColumnVector>           &o){
    /* The network starts with the 0th layer (input), followed by n_layers
        number of hidden layers, and then an output layer.
    */
    // Total number of coefficients in the model
    size_t N = model.u.size(); // assuming >= 1
    net.resize(N + 1);
    // o[k] is a vector of the output of the kth layer
    o.resize(N + 1);

    double (*activation)(const double&);
    if(model.activation==RELU)
        activation = &relu;
    else if(model.activation==SIGMOID)
        activation = &sigmoid;
    else
        activation = &tanh;

    o[0].resize(x.size() + 1);
    o[0] << 1.,x;

    for (size_t k = 1; k < N; k ++) {
        // o_k = activation(sum(o_{k-1} * u_{k-1}))
        // net_k just does the inner sum: input to the activation function
        net[k] = model.u[k-1].transpose() * o[k-1];
        o[k] = ColumnVector(model.u[k-1].cols() + 1);
        // This applies the activation function to give the actual node output
        o[k] << 1., net[k].unaryExpr(activation);
    }
    o[N] = model.u[N-1].transpose() * o[N-1];

    // Numerically stable calculation of softmax
    if(model.is_classification){
        double max_x = o[N].maxCoeff();
        o[N] = (o[N].array() - max_x).exp();
        o[N] /= o[N].sum();
    }
}

template <class Model, class Tuple>
void
MLP<Model, Tuple>::backPropogate(
        const ColumnVector                  &y_true,
        const ColumnVector                  &y_estimated,
        const std::vector<ColumnVector>     &net,
        const model_type                    &model,
        std::vector<ColumnVector>           &delta) {
    size_t N = model.u.size(); // assuming >= 1
    delta.resize(N);

    double (*activationDerivative)(const double&);
    if(model.activation==RELU)
        activationDerivative = &reluDerivative;
    else if(model.activation==SIGMOID)
        activationDerivative = &sigmoidDerivative;
    else
        activationDerivative = &tanhDerivative;

    delta.back() = y_estimated - y_true;
    for (size_t k = N - 1; k >= 1; k --) {
        // Do not include the bias terms
        delta[k-1] = model.u[k].bottomRows(model.u[k].rows()-1) * delta[k];
        delta[k-1] = delta[k-1].array() * net[k].unaryExpr(activationDerivative).array();
    }
}

} // namespace convex

} // namespace modules

} // namespace madlib

#endif


