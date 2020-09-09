/*
 * AC_AXES.C
 * 22-09-95
 * C.A.A.M. van der Linden
 * 26-10-04 ported to Matlab 6.5
 * O. Stroosma
 * syntax    : [ret,x0]=ac_axes(t,x,u,flag,axis)
 * parameters: ret : u in [wind/stability/body/earth]-axes
 *             axis: [0/1/2/3] for u in [wind/stability/body/earth]-axes
 * main prog : ac_mod.m  aero_mod.m
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

#define S_FUNCTION_NAME  ac_axes
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

/*
 * Defines for easy access of parameters which are passed in.
 */

//#define AXIS	ssGetArg(S,0)
#define AXIS	ssGetSFcnParam(S,0)

/*
 * Defines function to perform matrix multiplication y=Ax.
 */

void matmultiply(double A[3][3], double x[3] ,double y[3])
{
	int i,j;

	for (i=0;i<=2;i++) {
		y[i] = 0;
		for (j=0;j<=2;j++) {
			y[i] = y[i] + A[i][j]*x[j];
		}
	}
}


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

    ssSetNumSFcnParams(S, 1);  /* Number of expected parameters */
    if (ssGetNumSFcnParams(S) != ssGetSFcnParamsCount(S)) {
        /* Return if number of expected != number of actual parameters */
        return;
    }

    ssSetNumContStates(S, 0);
    ssSetNumDiscStates(S, 0);

    if (!ssSetNumInputPorts(S, 1)) return;
    ssSetInputPortWidth(S, 0, 15);
    ssSetInputPortRequiredContiguous(S, 0, true); /*direct input signal access*/
    /*
     * Set direct feedthrough flag (1=yes, 0=no).
     * A port has direct feedthrough if the input is used in either
     * the mdlOutputs or mdlGetTimeOfNextVarHit functions.
     * See matlabroot/simulink/src/sfuntmpl_directfeed.txt.
     */
    ssSetInputPortDirectFeedThrough(S, 0, 1);

    if (!ssSetNumOutputPorts(S, 1)) return;
    ssSetOutputPortWidth(S, 0, 12);

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
    //y[0] = u[0];

	double calp, cbet, cphi, cthe, cpsi;
	double salp, sbet, sphi, sthe, spsi;
	double Rsw[3][3], Rws[3][3], Rbs[3][3], Rsb[3][3];
	double Reb[3][3], Rbe[3][3];
	double ui[3], yw[3], ys[3], yb[3], ye[3];
	int i,j,axis;

	calp=cos(u[4]);
	cbet=cos(u[5]);
	cphi=cos(u[6]);
	cthe=cos(u[7]);
	cpsi=cos(u[8]);
	salp=sin(u[4]);
	sbet=sin(u[5]);
	sphi=sin(u[6]);
	sthe=sin(u[7]);
	spsi=sin(u[8]);

	/* define transformation matrices */
	/* wind versus stability axes */
	Rsw[0][0] =  cbet;
	Rsw[0][1] = -sbet;
	Rsw[0][2] =  0;
	Rsw[1][0] =  sbet;
	Rsw[1][1] =  cbet;
	Rsw[1][2] =  0;
	Rsw[2][0] =  0;
	Rsw[2][1] =  0;
	Rsw[2][2] =  1;

	for (i=0;i<=2;i++) {
		for (j=0;j<=2;j++) {
			Rws[i][j]=Rsw[j][i];
		}
	}

	/* stability versus body axes */
	Rbs[0][0] =  calp;
	Rbs[0][1] =  0;
	Rbs[0][2] = -salp;
	Rbs[1][0] =  0;
	Rbs[1][1] =  1;
	Rbs[1][2] =  0;
	Rbs[2][0] =  salp;
	Rbs[2][1] =  0;
	Rbs[2][2] =  calp;

	for (i=0;i<=2;i++) {
		for (j=0;j<=2;j++) {
			Rsb[i][j]=Rbs[j][i];
		}
	}

	/* body versus earth axes */
	Reb[0][0] =  cthe*cpsi;
	Reb[0][1] =  sphi*sthe*cpsi-cphi*spsi;
	Reb[0][2] =  cphi*sthe*cpsi+sphi*spsi;
	Reb[1][0] =  cthe*spsi;
	Reb[1][1] =  sphi*sthe*spsi+cphi*cpsi;
	Reb[1][2] =  cphi*sthe*spsi-sphi*cpsi;
	Reb[2][0] = -sthe;
	Reb[2][1] =  sphi*cthe;
	Reb[2][2] =  cphi*cthe;

	for (i=0;i<=2;i++) {
		for (j=0;j<=2;j++) {
			Rbe[i][j]=Reb[j][i];
		}
	}

	/* define vector to be transformed */
	for (i=0;i<=2;i++) {
		ui[i]=u[12+i];
	}

	/* perform transformations */
	axis = mxGetPr(AXIS)[0];

	if (axis==0) {
		/* ui in wind-axes */
		for (i=0;i<=2;i++) {
			yw[i] = ui[i];
		}

		matmultiply(Rsw,yw,ys);
		matmultiply(Rbs,ys,yb);
		matmultiply(Reb,yb,ye);
	}

	else if (axis==1) {
		/* ui in stability-axes */
		for (i=0;i<=2;i++) {
			ys[i] = ui[i];
		}

		matmultiply(Rws,ys,yw);
		matmultiply(Rbs,ys,yb);
		matmultiply(Reb,yb,ye);
	}

	else if (axis==2) {
		/* ui in body-axes */
		for (i=0;i<=2;i++) {
			yb[i] = ui[i];
		}

		matmultiply(Rsb,yb,ys);
		matmultiply(Reb,yb,ye);
		matmultiply(Rws,ys,yw);
	}

	else if (axis==3) {
		/* ui in earth-axes */
		for (i=0;i<=2;i++) {
			ye[i] = ui[i];
		}

		matmultiply(Rbe,ye,yb);
		matmultiply(Rsb,yb,ys);
		matmultiply(Rws,ys,yw);
	}

	for (i=0;i<=2;i++) {
		y[i]   = yw[i];
		y[3+i] = ys[i];
		y[6+i] = yb[i];
		y[9+i] = ye[i];
	}
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
