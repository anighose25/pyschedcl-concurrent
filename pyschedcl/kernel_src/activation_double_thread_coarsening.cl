// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// one of: [ TANH | RELU | LINEAR | SIGMOID | SCALEDTANH | ELU ]

#ifdef TANH
    #define ACTIVATION_FUNCTION(output) (tanh(output))
#elif defined SCALEDTANH
    #define ACTIVATION_FUNCTION(output) (1.7159f * tanh(0.66667f * output))
#elif SIGMOID
    #define ACTIVATION_FUNCTION(output) (1.0f / (1 + exp(-output)))
#elif defined RELU
    #define ACTIVATION_FUNCTION(output) (output> 0 ? output : 0)
#elif defined ELU
    #define ACTIVATION_FUNCTION(output) (output> 0 ? output : exp(output) - 1)
#elif defined LINEAR
    #define ACTIVATION_FUNCTION(output) (output)
#endif

#ifdef ACTIVATION_FUNCTION // protect against not defined

kernel void activate(const int N, global float *inout)
{

    const int globalId0 = get_group_id(0)*get_local_size(0)+(get_local_id(0)/32)*32*2 +get_local_id(0)%32+0*32;

    const int globalId1 = get_group_id(0)*get_local_size(0)+(get_local_id(0)/32)*32*2 +get_local_id(0)%32+1*32;


    inout[globalId0] = ACTIVATION_FUNCTION(inout[globalId0]); 
    inout[globalId1] = ACTIVATION_FUNCTION(inout[globalId1]);     
}
#endif
