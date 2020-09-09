/*
 * AC_ATMOS.C
 * 18-08-95
 * C.A.A.M. van der Linden
 * 26-10-04 ported to Matlab 6.5
 * O. Stroosma
 *
 * syntax    : [ret,x0]=ac_atmos(t,x,u,flag)
 * parameters: ret : [yatm,yad1,yad2]
 * main prog : ac_mod.m  eng_mod.m
 */

/*
 * sfuntmpl_basic.c: Basic 'C' template for a level 2 S-function.
 *
 *  -------------------------------------------------------------------------
 *  | See matlabroot/simulink/src/sfuntmpl_doc.c for a more detailed template |
 *  -------------------------------------------------------------------------
 *
 * Copyright 1990-2002 The MathWorks, Inc.
 * $Revision: 1.27 $
 */


/*
 * You must specify the S_FUNCTION_NAME as the name of your S-function
 * (i.e. replace sfuntmpl_basic with the name of your S-function).
 */

#define S_FUNCTION_NAME  ac_atmos
#define S_FUNCTION_LEVEL 2

/*
 * Need to include simstruc.h for the definition of the SimStruct and
 * its associated macro definitions.
 */
#include "simstruc.h"

/*
 * Need to include libraries for the definition of mathematical functions.
 */

#include <math.h>


/* Error handling
 * --------------
 *
 * You should use the following technique to report errors encountered within
 * an S-function:
 *
 *       ssSetErrorStatus(S,"Error encountered due to ...");
 *       return;
 *
 * Note that the 2nd argument to ssSetErrorStatus must be persistent memory.
 * It cannot be a local variable. For example the following will cause
 * unpredictable errors:
 *
 *      mdlOutputs()
 *      {
 *         char msg[256];         {ILLEGAL: to fix use "static char msg[256];"}
 *         sprintf(msg,"Error due to %s", string);
 *         ssSetErrorStatus(S,msg);
 *         return;
 *      }
 *
 * See matlabroot/simulink/src/sfuntmpl_doc.c for more details.
 */

/*====================*
 * S-function methods *
 *====================*/

/* Function: mdlInitializeSizes ===============================================
 * Abstract:
 *    The sizes information is used by Simulink to determine the S-function
 *    block's characteristics (number of inputs, outputs, states, etc.).
 */
static void mdlInitializeSizes(SimStruct *S)
{
    /* See sfuntmpl_doc.c for more details on the macros below */

    ssSetNumSFcnParams(S, 0);  /* Number of expected parameters */
    if (ssGetNumSFcnParams(S) != ssGetSFcnParamsCount(S)) {
        /* Return if number of expected != number of actual parameters */
        return;
    }

    ssSetNumContStates(S, 0);
    ssSetNumDiscStates(S, 0);

    if (!ssSetNumInputPorts(S, 1)) return;
    ssSetInputPortWidth(S, 0, 12);
    ssSetInputPortRequiredContiguous(S, 0, true); /*direct input signal access*/
    /*
     * Set direct feedthrough flag (1=yes, 0=no).
     * A port has direct feedthrough if the input is used in either
     * the mdlOutputs or mdlGetTimeOfNextVarHit functions.
     * See matlabroot/simulink/src/sfuntmpl_directfeed.txt.
     */
    ssSetInputPortDirectFeedThrough(S, 0, 1);

    if (!ssSetNumOutputPorts(S, 1)) return;
    ssSetOutputPortWidth(S, 0, 18);

    ssSetNumSampleTimes(S, 1);
    ssSetNumRWork(S, 0);
    ssSetNumIWork(S, 0);
    ssSetNumPWork(S, 0);
    ssSetNumModes(S, 0);
    ssSetNumNonsampledZCs(S, 0);

    ssSetOptions(S, 0);
}



/* Function: mdlInitializeSampleTimes =========================================
 * Abstract:
 *    This function is used to specify the sample time(s) for your
 *    S-function. You must register the same number of sample times as
 *    specified in ssSetNumSampleTimes.
 */
static void mdlInitializeSampleTimes(SimStruct *S)
{
    ssSetSampleTime(S, 0, CONTINUOUS_SAMPLE_TIME);
    ssSetOffsetTime(S, 0, 0.0);

}



//#define MDL_INITIALIZE_CONDITIONS   /* Change to #undef to remove function */
#undef MDL_INITIALIZE_CONDITIONS   /* Change to #undef to remove function */
#if defined(MDL_INITIALIZE_CONDITIONS)
  /* Function: mdlInitializeConditions ========================================
   * Abstract:
   *    In this function, you should initialize the continuous and discrete
   *    states for your S-function block.  The initial states are placed
   *    in the state vector, ssGetContStates(S) or ssGetRealDiscStates(S).
   *    You can also perform any other initialization activities that your
   *    S-function may require. Note, this routine will be called at the
   *    start of simulation and if it is present in an enabled subsystem
   *    configured to reset states, it will be call when the enabled subsystem
   *    restarts execution to reset the states.
   */
  static void mdlInitializeConditions(SimStruct *S)
  {
  }
#endif /* MDL_INITIALIZE_CONDITIONS */



//#define MDL_START  /* Change to #undef to remove function */
#undef MDL_START  /* Change to #undef to remove function */
#if defined(MDL_START) 
  /* Function: mdlStart =======================================================
   * Abstract:
   *    This function is called once at start of model execution. If you
   *    have states that should be initialized once, this is the place
   *    to do it.
   */
  static void mdlStart(SimStruct *S)
  {
  }
#endif /*  MDL_START */



/* Function: mdlOutputs =======================================================
 * Abstract:
 *    In this function, you compute the outputs of your S-function
 *    block. Generally outputs are placed in the output vector, ssGetY(S).
 */
static void mdlOutputs(SimStruct *S, int_T tid)
{
    const real_T *u = (const real_T*) ssGetInputPortSignal(S,0);
    real_T       *y = ssGetOutputPortSignal(S,0);

    /* define global variables */
    static double g0     =  9.80665;
    static double T0     =  288.15;
    static double rho0   =  1.225;
    static double p0     =  101325;
    static double lambda = -0.0065;
    static double GASCON =  287.05;
    static double gammah =  1.4;
    static double RADIUS =  6371020;
    static double Hs     =  11000;

	double he, Hgeo, T, pa, rho, g, Vsound, VTAS, M, qdyn,
	       mu, Reynl, qrel, qc, ptot, Ttot, VEAS, VCAS, VIAS;

	VTAS = u[3];
	he   = u[9];


	/* atmospheric data */
	Hgeo   = RADIUS*he/(RADIUS+he);

  	if (Hgeo<=Hs) {
		T   = T0+lambda*Hgeo;
		rho = rho0*pow((T/T0),-(g0/(GASCON*lambda)+1));
	}
	else {
		T   = T0+lambda*Hs;
		rho = rho0*pow((T/T0),-(g0/(GASCON*lambda)+1));
		rho = rho*exp(-g0/(GASCON*T)*(Hgeo-Hs));
	}

	pa     = rho*GASCON*T;
	g      = g0*(RADIUS*RADIUS)/((RADIUS+he)*(RADIUS+he));
	Vsound = sqrt(gammah*GASCON*T);

	y[0] = pa;        /* ambient pressure */
	y[1] = rho;       /* air density */
	y[2] = T;         /* ambient temperature */
	y[3] = g;         /* acceleration of gravity */
	y[4] = he;        /* pressure altitude (geometrical altitude) */
	y[5] = he;        /* radio altitude (geometrical altitude) */
	y[6] = Hgeo;      /* geopotential altitude */
	y[7] = Vsound;    /* speed of sound */


	/* airdata 1 */
	M      = VTAS/Vsound;
	qdyn   = 0.5*rho*VTAS*VTAS;

	y[8] = M;         /* Mach number */
	y[9] = qdyn;      /* dynamic pressure */


	/* airdata 2 */
	mu    = 1.458e-6*pow(T,1.5)/(T+110.4);
	Reynl = rho*VTAS/mu;

	qrel = pow((1+0.2*M*M),3.5)-1;
	qc   = pa*qrel;
	ptot = pa+qc;
	Ttot = T*(1+0.2*M*M);

	VEAS = sqrt(2*qdyn/rho0);
	VCAS = sqrt(2*3.5*p0/rho0*(pow((1+qc/p0),(1/3.5))-1));
	VIAS = VCAS;
	
	y[10] = Reynl;    /* Reynolds number per unit length */
	y[11] = qc;       /* impact pressure */
	y[12] = qrel;     /* relative impact pressure */
	y[13] = ptot;     /* total pressure */
	y[14] = Ttot;     /* total temperature */
	y[15] = VEAS;     /* equivalent airspeed */
	y[16] = VCAS;     /* calibrated airspeed */
	y[17] = VIAS;     /* indicated airspeed */

}



//#define MDL_UPDATE  /* Change to #undef to remove function */
#undef MDL_UPDATE  /* Change to #undef to remove function */
#if defined(MDL_UPDATE)
  /* Function: mdlUpdate ======================================================
   * Abstract:
   *    This function is called once for every major integration time step.
   *    Discrete states are typically updated here, but this function is useful
   *    for performing any tasks that should only take place once per
   *    integration step.
   */
  static void mdlUpdate(SimStruct *S, int_T tid)
  {
  }
#endif /* MDL_UPDATE */



//#define MDL_DERIVATIVES  /* Change to #undef to remove function */
#undef MDL_DERIVATIVES  /* Change to #undef to remove function */
#if defined(MDL_DERIVATIVES)
  /* Function: mdlDerivatives =================================================
   * Abstract:
   *    In this function, you compute the S-function block's derivatives.
   *    The derivatives are placed in the derivative vector, ssGetdX(S).
   */
  static void mdlDerivatives(SimStruct *S)
  {
  }
#endif /* MDL_DERIVATIVES */



/* Function: mdlTerminate =====================================================
 * Abstract:
 *    In this function, you should perform any actions that are necessary
 *    at the termination of a simulation.  For example, if memory was
 *    allocated in mdlStart, this is the place to free it.
 */
static void mdlTerminate(SimStruct *S)
{
}


/*======================================================*
 * See sfuntmpl_doc.c for the optional S-function methods *
 *======================================================*/

/*=============================*
 * Required S-function trailer *
 *=============================*/

#ifdef  MATLAB_MEX_FILE    /* Is this file being compiled as a MEX-file? */
#include "simulink.c"      /* MEX-file interface mechanism */
#else
#include "cg_sfun.h"       /* Code generation registration function */
#endif
