/* File: swig.i */
%module citation

%{
    #define SWIG_FILE_WITH_INIT
    #include "c_citation.h"
%}

%include "typemaps.i"
%include "numpy.i"
%init %{
    import_array();
%}

#include "rtwtypes.h"
%rename(terminate) c_citation_terminate;
extern void initialize(void);
extern void c_citation_terminate(void);

%apply(double IN_ARRAY1[ANY]){(real_T cmd[10])}
%apply(double ARGOUT_ARRAY1[ANY]){(real_T output_states[12])}


%inline %{
    extern void step(real_T cmd[10], real_T output_states[12]);
%}

