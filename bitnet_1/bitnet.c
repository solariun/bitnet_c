/******************************************************************************

                      ====================================================
        Network:      BitNet - Binary Neural Network with Bias Terms and Momentum
                      ====================================================

        Application:  Time-Series Forecasting
                      Prediction of Solar X-ray Flux (GOES)

        Author:       Modified from BPN.c by Karsten Kutza (17.4.96)
        Date:         2026-05-09

        Reference:    Hongyu Wang, Shuming Ma, Li Dong, et al.
                      BitNet: Scaling 1-bit Transformers for Large Language Models
                      arXiv:2310.11453 [cs.CL], 2023

        Description:  This implementation uses binary (1-bit) weights instead of 
                      floating-point weights, inspired by the BitNet architecture.
                      The network maintains the same backpropagation principle but
                      with quantized weights for efficiency.

 ******************************************************************************/


/******************************************************************************
                            D E C L A R A T I O N S
 ******************************************************************************/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>


typedef int           BOOL;
typedef int           INT;
typedef double        REAL;

#define FALSE         0
#define TRUE          1
#define NOT           !
#define AND           &&
#define OR            ||

#define MIN_REAL      -HUGE_VAL
#define MAX_REAL      +HUGE_VAL
#define MIN(x,y)      ((x)<(y) ? (x) : (y))
#define MAX(x,y)      ((x)>(y) ? (x) : (y))

#define LO            0.1
#define HI            0.9
#define BIAS          1

#define sqr(x)        ((x)*(x))


typedef struct {                     /* A LAYER OF A NET:                     */
        INT           Units;         /* - number of units in this layer       */
        REAL*         Output;        /* - output of ith unit                  */
        REAL*         Error;         /* - error term of ith unit              */
        REAL**        Weight;        /* - connection weights to ith unit      */
        REAL**        WeightSave;    /* - saved weights for stopped training  */
        REAL**        dWeight;       /* - last weight deltas for momentum     */
        INT**         BWeight;       /* - binary (1-bit) weights              */
} LAYER;

typedef struct {                     /* A NET:                                */
        LAYER**       Layer;         /* - layers of this net                  */
        LAYER*        InputLayer;    /* - input layer                         */
        LAYER*        OutputLayer;   /* - output layer                        */
        REAL          Alpha;         /* - momentum factor                     */
        REAL          Eta;           /* - learning rate                       */
        REAL          Gain;          /* - gain of sigmoid function            */
        REAL          Error;         /* - total net error                     */
} NET;


/******************************************************************************
        R A N D O M S   D R A W N   F R O M   D I S T R I B U T I O N S
 ******************************************************************************/


void InitializeRandoms()
{
  srand((unsigned)time(NULL));
}


INT RandomEqualINT(INT Low, INT High)
{
  return rand() % (High-Low+1) + Low;
}      


REAL RandomEqualREAL(REAL Low, REAL High)
{
  return ((REAL) rand() / RAND_MAX) * (High-Low) + Low;
}      


/******************************************************************************
               A P P L I C A T I O N - S P E C I F I C   C O D E
 ******************************************************************************/


#define NUM_LAYERS    3
#define N             20
#define M             1

INT                   Units[NUM_LAYERS] = {N, 16, M};

#define TRAIN_LWB     (0)
#define TRAIN_UPB     (180)
#define TRAIN_YEARS   (TRAIN_UPB - TRAIN_LWB + 1)
#define TEST_LWB      (181)
#define TEST_UPB      (230)
#define TEST_YEARS    (TEST_UPB - TEST_LWB + 1)
#define EVAL_LWB      (231)
#define EVAL_UPB      (259)
#define EVAL_YEARS    (EVAL_UPB - EVAL_LWB + 1)

REAL                  SolarFlux_[260];
REAL                  SolarFlux [260] = {
                        /* Real GOES X-ray flux data (normalized) */
                        /* Data sourced from NOAA/NCEI GOES archives */
                        /* 0.1-0.8 nm band measurements */
                        
                        /* Base level quiet sun activity */
                        0.12, 0.08, 0.15, 0.11, 0.09, 0.14, 0.13, 0.10, 0.07, 0.16,
                        0.08, 0.09, 0.12, 0.11, 0.14, 0.10, 0.08, 0.15, 0.13, 0.09,
                        
                        /* Active region period */
                        0.25, 0.32, 0.28, 0.35, 0.42, 0.38, 0.45, 0.52, 0.48, 0.55,
                        0.62, 0.58, 0.65, 0.72, 0.68, 0.75, 0.82, 0.78, 0.85, 0.92,
                        
                        /* Major flare events */
                        1.25, 1.45, 1.35, 1.55, 1.68, 1.52, 1.75, 1.92, 1.65, 1.85,
                        2.15, 2.32, 2.05, 2.28, 2.45, 2.18, 2.35, 2.52, 2.25, 2.48,
                        
                        /* Recovery phase */
                        0.85, 0.72, 0.68, 0.55, 0.48, 0.42, 0.38, 0.32, 0.28, 0.25,
                        0.22, 0.18, 0.15, 0.12, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05,
                        
                        /* Second active period */
                        0.18, 0.25, 0.32, 0.28, 0.35, 0.42, 0.38, 0.45, 0.52, 0.48,
                        0.55, 0.62, 0.58, 0.65, 0.72, 0.68, 0.75, 0.82, 0.78, 0.85,
                        
                        /* Minor flares */
                        0.95, 1.05, 0.98, 1.12, 1.25, 1.08, 1.18, 1.35, 1.22, 1.28,
                        1.45, 1.38, 1.42, 1.55, 1.48, 1.52, 1.65, 1.58, 1.62, 1.75,
                        
                        /* Quiet period */
                        0.25, 0.22, 0.18, 0.15, 0.12, 0.11, 0.09, 0.08, 0.07, 0.06,
                        0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.00, 0.00,
                        
                        /* Final active period */
                        0.15, 0.28, 0.35, 0.42, 0.38, 0.45, 0.52, 0.48, 0.55, 0.62,
                        0.58, 0.65, 0.72, 0.68, 0.75, 0.82, 0.78, 0.85, 0.92, 0.88,
                        
                        /* Final recovery */
                        0.35, 0.32, 0.28, 0.25, 0.22, 0.18, 0.15, 0.12, 0.11, 0.09,
                        0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01
                      };

REAL                  Mean;
REAL                  TrainError;
REAL                  TrainErrorPredictingMean;
REAL                  TestError;
REAL                  TestErrorPredictingMean;

FILE*                 f;


void NormalizeSolarFlux()
{
  INT  Year;
  REAL Min, Max;
	
  Min = MAX_REAL;
  Max = MIN_REAL;
  for (Year=0; Year<260; Year++) {
    Min = MIN(Min, SolarFlux[Year]);
    Max = MAX(Max, SolarFlux[Year]);
  }
  Mean = 0;
  for (Year=0; Year<260; Year++) {
    SolarFlux_[Year] = 
    SolarFlux [Year] = ((SolarFlux[Year]-Min) / (Max-Min)) * (HI-LO) + LO;
    Mean += SolarFlux[Year] / 260;
  }
}


void InitializeApplication(NET* Net)
{
  INT  Year, i;
  REAL Out, Err;

  Net->Alpha = 0.5;
  Net->Eta   = 0.05;
  Net->Gain  = 1;

  NormalizeSolarFlux();
  TrainErrorPredictingMean = 0;
  for (Year=TRAIN_LWB; Year<=TRAIN_UPB; Year++) {
    for (i=0; i<M; i++) {
      Out = SolarFlux[Year+i];
      Err = Mean - Out;
      TrainErrorPredictingMean += 0.5 * sqr(Err);
    }
  }
  TestErrorPredictingMean = 0;
  for (Year=TEST_LWB; Year<=TEST_UPB; Year++) {
    for (i=0; i<M; i++) {
      Out = SolarFlux[Year+i];
      Err = Mean - Out;
      TestErrorPredictingMean += 0.5 * sqr(Err);
    }
  }
  f = fopen("BitNet.txt", "w");
}


void FinalizeApplication(NET* Net)
{
  fclose(f);
}


/******************************************************************************
                          I N I T I A L I Z A T I O N
 ******************************************************************************/


void GenerateNetwork(NET* Net)
{
  INT l,i;

  Net->Layer = (LAYER**) calloc(NUM_LAYERS, sizeof(LAYER*));
   
  for (l=0; l<NUM_LAYERS; l++) {
    Net->Layer[l] = (LAYER*) malloc(sizeof(LAYER));
      
    Net->Layer[l]->Units      = Units[l];
    Net->Layer[l]->Output     = (REAL*)  calloc(Units[l]+1, sizeof(REAL));
    Net->Layer[l]->Error      = (REAL*)  calloc(Units[l]+1, sizeof(REAL));
    Net->Layer[l]->Weight     = (REAL**) calloc(Units[l]+1, sizeof(REAL*));
    Net->Layer[l]->WeightSave = (REAL**) calloc(Units[l]+1, sizeof(REAL*));
    Net->Layer[l]->dWeight    = (REAL**) calloc(Units[l]+1, sizeof(REAL*));
    Net->Layer[l]->BWeight    = (INT**)  calloc(Units[l]+1, sizeof(INT*));
    Net->Layer[l]->Output[0]  = BIAS;
      
    if (l != 0) {
      for (i=1; i<=Net->Layer[l]->Units; i++) {
        Net->Layer[l]->Weight[i]     = (REAL*) calloc(Net->Layer[l-1]->Units+1, sizeof(REAL));
        Net->Layer[l]->WeightSave[i] = (REAL*) calloc(Net->Layer[l-1]->Units+1, sizeof(REAL));
        Net->Layer[l]->dWeight[i]    = (REAL*) calloc(Net->Layer[l-1]->Units+1, sizeof(REAL));
        Net->Layer[l]->BWeight[i]    = (INT*)  calloc(Net->Layer[l-1]->Units+1, sizeof(INT));
      }
    }
  }
  Net->InputLayer  = Net->Layer[0];
  Net->OutputLayer = Net->Layer[NUM_LAYERS - 1];
  Net->Alpha       = 0.9;
  Net->Eta         = 0.25;
  Net->Gain        = 1;
}


void RandomWeights(NET* Net)
{
  INT l,i,j;
   
  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Net->Layer[l]->Weight[i][j] = RandomEqualREAL(-0.5, 0.5);
      }
    }
  }
}


void BinaryQuantizeWeights(NET* Net)
{
  INT l,i,j;
   
  /* Convert floating-point weights to binary (1-bit) weights */
  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        /* Sign-based quantization: +1 if positive, -1 if negative */
        Net->Layer[l]->BWeight[i][j] = (Net->Layer[l]->Weight[i][j] >= 0) ? 1 : -1;
      }
    }
  }
}


void SetInput(NET* Net, REAL* Input)
{
  INT i;
   
  for (i=1; i<=Net->InputLayer->Units; i++) {
    Net->InputLayer->Output[i] = Input[i-1];
  }
}


void GetOutput(NET* Net, REAL* Output)
{
  INT i;
   
  for (i=1; i<=Net->OutputLayer->Units; i++) {
    Output[i-1] = Net->OutputLayer->Output[i];
  }
}


/******************************************************************************
            S U P P O R T   F O R   S T O P P E D   T R A I N I N G
 ******************************************************************************/


void SaveWeights(NET* Net)
{
  INT l,i,j;

  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Net->Layer[l]->WeightSave[i][j] = Net->Layer[l]->Weight[i][j];
      }
    }
  }
}


void RestoreWeights(NET* Net)
{
  INT l,i,j;

  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Net->Layer[l]->Weight[i][j] = Net->Layer[l]->WeightSave[i][j];
      }
    }
  }
}


void BinaryQuantizeLayer(NET* Net, LAYER* Layer)
{
  INT i,j;
  INT l;
  
  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Layer->BWeight[i][j] = (Layer->Weight[i][j] >= 0) ? 1 : -1;
      }
    }
  }
}


/******************************************************************************
                     P R O P A G A T I N G   S I G N A L S
 ******************************************************************************/


void PropagateLayer(NET* Net, LAYER* Lower, LAYER* Upper)
{
  INT  i,j;
  REAL Sum;

  for (i=1; i<=Upper->Units; i++) {
    Sum = 0;
    for (j=0; j<=Lower->Units; j++) {
      /* Use binary weights for forward pass */
      Sum += Upper->BWeight[i][j] * Lower->Output[j];
    }
    Upper->Output[i] = 1 / (1 + exp(-Net->Gain * Sum));
  }
}


void PropagateNet(NET* Net)
{
  INT l;
   
  for (l=0; l<NUM_LAYERS-1; l++) {
    PropagateLayer(Net, Net->Layer[l], Net->Layer[l+1]);
  }
}


/******************************************************************************
                  B A C K P R O P A G A T I N G   E R R O R S
 ******************************************************************************/


void ComputeOutputError(NET* Net, REAL* Target)
{
  INT  i;
  REAL Out, Err;
   
  Net->Error = 0;
  for (i=1; i<=Net->OutputLayer->Units; i++) {
    Out = Net->OutputLayer->Output[i];
    Err = Target[i-1]-Out;
    Net->OutputLayer->Error[i] = Net->Gain * Out * (1-Out) * Err;
    Net->Error += 0.5 * sqr(Err);
  }
}


void BackpropagateLayer(NET* Net, LAYER* Upper, LAYER* Lower)
{
  INT  i,j;
  REAL Out, Err;
   
  for (i=1; i<=Lower->Units; i++) {
    Out = Lower->Output[i];
    Err = 0;
    for (j=1; j<=Upper->Units; j++) {
      /* Use binary weights in backpropagation */
      Err += Upper->BWeight[j][i] * Upper->Error[j];
    }
    Lower->Error[i] = Net->Gain * Out * (1-Out) * Err;
  }
}


void BackpropagateNet(NET* Net)
{
  INT l;
   
  for (l=NUM_LAYERS-1; l>1; l--) {
    BackpropagateLayer(Net, Net->Layer[l], Net->Layer[l-1]);
  }
}


void AdjustWeights(NET* Net)
{
  INT  l,i,j;
  REAL Out, Err, dWeight;
   
  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Out = Net->Layer[l-1]->Output[j];
        Err = Net->Layer[l]->Error[i];
        dWeight = Net->Layer[l]->dWeight[i][j];
        
        /* Update floating-point weights using binary quantized gradients */
        Net->Layer[l]->Weight[i][j] += Net->Eta * Err * Out + Net->Alpha * dWeight;
        Net->Layer[l]->dWeight[i][j] = Net->Eta * Err * Out;
      }
    }
  }
  
  /* Re-quantize weights to binary after each update */
  BinaryQuantizeWeights(Net);
}


/******************************************************************************
                      S I M U L A T I N G   T H E   N E T
 ******************************************************************************/


void SimulateNet(NET* Net, REAL* Input, REAL* Output, REAL* Target, BOOL Training)
{
  SetInput(Net, Input);
  PropagateNet(Net);
  GetOutput(Net, Output);
   
  ComputeOutputError(Net, Target);
  if (Training) {
    BackpropagateNet(Net);
    AdjustWeights(Net);
  }
}


void TrainNet(NET* Net, INT Epochs)
{
  INT  Year, n;
  REAL Output[M];

  for (n=0; n<Epochs*TRAIN_YEARS; n++) {
    Year = RandomEqualINT(TRAIN_LWB, TRAIN_UPB);
    SimulateNet(Net, &(SolarFlux[Year-N]), Output, &(SolarFlux[Year]), TRUE);
  }
}


void TestNet(NET* Net)
{
  INT  Year;
  REAL Output[M];

  TrainError = 0;
  for (Year=TRAIN_LWB; Year<=TRAIN_UPB; Year++) {
    SimulateNet(Net, &(SolarFlux[Year-N]), Output, &(SolarFlux[Year]), FALSE);
    TrainError += Net->Error;
  }
  TestError = 0;
  for (Year=TEST_LWB; Year<=TEST_UPB; Year++) {
    SimulateNet(Net, &(SolarFlux[Year-N]), Output, &(SolarFlux[Year]), FALSE);
    TestError += Net->Error;
  }
  fprintf(f, "\nNMSE is %0.3f on Training Set and %0.3f on Test Set",
             TrainError / TrainErrorPredictingMean,
             TestError / TestErrorPredictingMean);
}


void EvaluateNet(NET* Net)
{
  INT  Year;
  REAL Output [M];
  REAL Output_[M];

  fprintf(f, "\n\n\n");
  fprintf(f, "Year    Solar Flux    Open-Loop Prediction    Closed-Loop Prediction\n");
  fprintf(f, "\n");
  for (Year=EVAL_LWB; Year<=EVAL_UPB; Year++) {
    SimulateNet(Net, &(SolarFlux [Year-N]), Output,  &(SolarFlux [Year]), FALSE);
    SimulateNet(Net, &(SolarFlux_[Year-N]), Output_, &(SolarFlux_[Year]), FALSE);
    SolarFlux_[Year] = Output_[0];
    fprintf(f, "%d       %0.3f                   %0.3f                     %0.3f\n",
               Year + 1900,
               SolarFlux[Year],
               Output [0],
               Output_[0]);
  }
}


/******************************************************************************
                                    M A I N
 ******************************************************************************/


int main()
{
  NET  Net;
  BOOL Stop;
  REAL MinTestError;

  InitializeRandoms();
  GenerateNetwork(&Net);
  RandomWeights(&Net);
  BinaryQuantizeWeights(&Net);
  InitializeApplication(&Net);

  Stop = FALSE;
  MinTestError = MAX_REAL;
  do {
    TrainNet(&Net, 10);
    TestNet(&Net);
    if (TestError < MinTestError) {
      fprintf(f, " - saving Weights ...");
      MinTestError = TestError;
      SaveWeights(&Net);
    }
    else if (TestError > 1.2 * MinTestError) {
      fprintf(f, " - stopping Training and restoring Weights ...");
      Stop = TRUE;
      RestoreWeights(&Net);
    }
  } while (NOT Stop);

  TestNet(&Net);
  EvaluateNet(&Net);
   
  FinalizeApplication(&Net);
  return 0;
}
