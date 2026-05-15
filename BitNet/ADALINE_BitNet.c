/******************************************************************************

                      ===============
        Network:      Adaline Network (BitNet Ternary Quantized)
                      ===============

        Application:  Pattern Recognition
                      Classification of Digits 0-9

        Author:       Adapted from Karsten Kutza (1996) for BitNet ternary quantization
        Date:         2026-05-15

        Reference:    B. Widrow, M.E. Hoff
                      Adaptive Switching Circuits
                      1960 IRE WESCON Convention Record, IRE, New York, NY,
                      pp. 96-104, 1960
                      BitNet b1.58: Ternary Weights {-1, 0, +1} with Absmean Scaling (Wang et al., 2023)

 ******************************************************************************/


/******************************************************************************
                            D E C L A R A T I O N S
 ******************************************************************************/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>


typedef int           BOOL;
typedef char          CHAR;
typedef int           INT;

#define FALSE         0
#define TRUE          1
#define NOT           !
#define AND           &&
#define OR            ||

#define MIN(x,y)      ((x)<(y) ? (x) : (y))
#define MAX(x,y)      ((x)>(y) ? (x) : (y))

#define LO            -1
#define HI            +1
#define BIAS           1

#define sqr(x)        ((x)*(x))


typedef struct {                     /* A LAYER OF A NET:                     */
        INT           Units;         /* - number of units in this layer       */
        INT*          Output;        /* - output of ith unit (ternary: -1,0,1) */
        INT**         Weight;        /* - connection weights to ith unit (ternary) */
        REAL          Scale;         /* - absmean scaling factor for this layer */
} LAYER;

typedef struct {                     /* A NET:                                */
        LAYER*        InputLayer;    /* - input layer                         */
        LAYER*        OutputLayer;   /* - output layer                        */
        REAL          Eta;           /* - learning rate                       */
        REAL          Error;         /* - total net error                     */
        REAL          Epsilon;       /* - net error to terminate training     */
} NET;


/******************************************************************************
        R A N D O M S   D R A W N   F R O M   D I S T R I B U T I O N S
 ******************************************************************************/


void InitializeRandoms()
{
  srand(4711);
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


#define NUM_DATA      10
#define X             5
#define Y             7

#define N             (X * Y)
#define M             10

CHAR                  Pattern[NUM_DATA][Y][X] = { { " OOO ",
                                                    "O   O",
                                                    "O   O",
                                                    "O   O",
                                                    "O   O",
                                                    "O   O",
                                                    " OOO "  },

                                                  { "  O  ",
                                                    " OO  ",
                                                    "O O  ",
                                                    "  O  ",
                                                    "  O  ",
                                                    "  O  ",
                                                    "  O  "  },

                                                  { " OOO ",
                                                    "O   O",
                                                    "    O",
                                                    "   O ",
                                                    "  O  ",
                                                    " O   ",
                                                    "OOOOO"  },

                                                  { " OOO ",
                                                    "O   O",
                                                    "    O",
                                                    " OOO ",
                                                    "    O",
                                                    "O   O",
                                                    " OOO "  },

                                                  { "   O ",
                                                    "  OO ",
                                                    " O O ",
                                                    "O  O ",
                                                    "OOOOO",
                                                    "   O ",
                                                    "   O "  },

                                                  { "OOOOO",
                                                    "O    ",
                                                    "O    ",
                                                    "OOOO ",
                                                    "    O",
                                                    "O   O",
                                                    " OOO "  },

                                                  { " OOO ",
                                                    "O   O",
                                                    "O    ",
                                                    "OOOO ",
                                                    "O   O",
                                                    "O   O",
                                                    " OOO "  },

                                                  { "OOOOO",
                                                    "    O",
                                                    "    O",
                                                    "   O ",
                                                    "  O  ",
                                                    " O   ",
                                                    "O    "  },

                                                  { " OOO ",
                                                    "O   O",
                                                    "O   O",
                                                    " OOO ",
                                                    "O   O",
                                                    "O   O",
                                                    " OOO "  },

                                                  { " OOO ",
                                                    "O   O",
                                                    "O   O",
                                                    " OOOO",
                                                    "    O",
                                                    "O   O",
                                                    " OOO "  } };

INT                   Input [NUM_DATA][N];
INT                   Output[NUM_DATA][M] =

                                  { {HI, LO, LO, LO, LO, LO, LO, LO, LO, LO},
                                    {LO, HI, LO, LO, LO, LO, LO, LO, LO, LO},
                                    {LO, LO, HI, LO, LO, LO, LO, LO, LO, LO},
                                    {LO, LO, LO, HI, LO, LO, LO, LO, LO, LO},
                                    {LO, LO, LO, LO, HI, LO, LO, LO, LO, LO},
                                    {LO, LO, LO, LO, LO, HI, LO, LO, LO, LO},
                                    {LO, LO, LO, LO, LO, LO, HI, LO, LO, LO},
                                    {LO, LO, LO, LO, LO, LO, LO, HI, LO, LO},
                                    {LO, LO, LO, LO, LO, LO, LO, LO, HI, LO},
                                    {LO, LO, LO, LO, LO, LO, LO, LO, LO, HI}  };

FILE*                 f;


void InitializeApplication(NET* Net)
{
  INT n,i,j;

  Net->Eta     = 0.001;
  Net->Epsilon = 0.0001;

  for (n=0; n<NUM_DATA; n++) {
    for (i=0; i<Y; i++) {
      for (j=0; j<X; j++) {
        Input[n][i*X+j] = (Pattern[n][i][j] == 'O') ? HI : LO;
      }
    }
  }
  f = fopen("ADALINE_BitNet.txt", "w");
}


void WriteInput(NET* Net, INT* Input)
{
  INT i;
  
  for (i=0; i<N; i++) {
    if (i%X == 0) {
      fprintf(f, "\n");
    }
    fprintf(f, "%c", (Input[i] == HI) ? 'O' : ' ');
  }
  fprintf(f, " -> ");
}


void WriteOutput(NET* Net, INT* Output)
{
  INT i;
  INT Count, Index;
  
  Count = 0;
  for (i=0; i<M; i++) {
    if (Output[i] == HI) {
      Count++;
      Index = i;
    }
  }
  if (Count == 1)
    fprintf(f, "%i\n", Index);
  else
    fprintf(f, "%s\n", "invalid");
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
  INT i;

  Net->InputLayer  = (LAYER*) malloc(sizeof(LAYER));
  Net->OutputLayer = (LAYER*) malloc(sizeof(LAYER));

  Net->InputLayer->Units       = N;
  Net->InputLayer->Output      = (INT*)   calloc(N+1, sizeof(INT));
  Net->InputLayer->Output[0]   = BIAS;
  Net->InputLayer->Weight      = NULL; // Not used - input is fixed ternary
  Net->InputLayer->Scale       = 1.0;  // Identity scale

  Net->OutputLayer->Units      = M;
  Net->OutputLayer->Output     = (INT*)   calloc(M+1, sizeof(INT));
  Net->OutputLayer->Weight     = (INT**) calloc(M+1, sizeof(INT*));
  Net->OutputLayer->Scale      = 0.0;  // Will be computed after random init

  for (i=1; i<=M; i++) {
    Net->OutputLayer->Weight[i] = (INT*) calloc(N+1, sizeof(INT));
  }

  Net->Eta     = 0.1;
  Net->Epsilon = 0.01;
}


void RandomWeights(NET* Net)
{
  INT i,j;
  REAL r;
  
  // Initialize real-valued weights in [-0.5, 0.5], then quantize to ternary
  for (i=1; i<=Net->OutputLayer->Units; i++) {
    for (j=0; j<=Net->InputLayer->Units; j++) {
      r = RandomEqualREAL(-0.5, 0.5);
      // Store as real temporarily for quantization
      Net->OutputLayer->Weight[i][j] = (INT) round(r * 2); // Map [-0.5,0.5] -> {-1,0,1}
    }
  }
  
  // Compute absmean scale for output layer weights
  REAL sum_abs = 0.0;
  INT count = 0;
  for (i=1; i<=Net->OutputLayer->Units; i++) {
    for (j=0; j<=Net->InputLayer->Units; j++) {
      sum_abs += abs(Net->OutputLayer->Weight[i][j]);
      count++;
    }
  }
  Net->OutputLayer->Scale = sum_abs / count;
}


void SetInput(NET* Net, INT* Input, BOOL Protocoling)
{
  INT i;
  
  for (i=1; i<=Net->InputLayer->Units; i++) {
    Net->InputLayer->Output[i] = Input[i-1];
  }
  if (Protocoling) {
    WriteInput(Net, Input);
  }
}


void GetOutput(NET* Net, INT* Output, BOOL Protocoling)
{
  INT i;
  
  for (i=1; i<=Net->OutputLayer->Units; i++) {
    Output[i-1] = Net->OutputLayer->Output[i];
  }
  if (Protocoling) {
    WriteOutput(Net, Output);
  }
}


/******************************************************************************
                     P R O P A G A T I N G   S I G N A L S
 ******************************************************************************/


void PropagateNet(NET* Net)
{
  INT  i,j;
  REAL Sum;
  
  for (i=1; i<=Net->OutputLayer->Units; i++) {
    Sum = 0.0;
    // Ternary dot product: use conditional adds/subtracts instead of multiplies
    for (j=0; j<=Net->InputLayer->Units; j++) {
      INT w = Net->OutputLayer->Weight[i][j];
      INT x = Net->InputLayer->Output[j];
      if (w == 1) Sum += x;
      else if (w == -1) Sum -= x;
      // if w==0, do nothing
    }
    Net->OutputLayer->Activation[i] = Sum;
    if (Sum >= 0)
      Net->OutputLayer->Output[i] = HI;
    else
      Net->OutputLayer->Output[i] = LO;
  }
}


/******************************************************************************
                       A D J U S T I N G   W E I G H T S
 ******************************************************************************/


void ComputeOutputError(NET* Net, INT* Target)
{
  INT  i;
  REAL Err;
  
  Net->Error = 0;
  for (i=1; i<=Net->OutputLayer->Units; i++) {
    Err = Target[i-1] - Net->OutputLayer->Activation[i];
    Net->OutputLayer->Error[i] = Err;
    Net->Error += 0.5 * sqr(Err);
  }
}


void AdjustWeights(NET* Net)
{
  INT  i,j;
  INT  Out;
  REAL Err;
  
  for (i=1; i<=Net->OutputLayer->Units; i++) {
    for (j=0; j<=Net->InputLayer->Units; j++) {
      Out = Net->InputLayer->Output[j];
      Err = Net->OutputLayer->Error[i];
      // Update real-valued delta, then quantize back to ternary
      REAL delta = Net->Eta * Err * Out;
      REAL new_w_real = (REAL)Net->OutputLayer->Weight[i][j] + delta;
      
      // Quantize new weight using absmean scaling
      if (new_w_real >= Net->OutputLayer->Scale)
        Net->OutputLayer->Weight[i][j] = HI;
      else if (new_w_real <= -Net->OutputLayer->Scale)
        Net->OutputLayer->Weight[i][j] = LO;
      else
        Net->OutputLayer->Weight[i][j] = 0;
    }
  }
  
  // Recompute scale after weight updates
  REAL sum_abs = 0.0;
  INT count = 0;
  for (i=1; i<=Net->OutputLayer->Units; i++) {
    for (j=0; j<=Net->InputLayer->Units; j++) {
      sum_abs += abs(Net->OutputLayer->Weight[i][j]);
      count++;
    }
  }
  Net->OutputLayer->Scale = sum_abs / count;
}


/******************************************************************************
                      S I M U L A T I N G   T H E   N E T
 ******************************************************************************/


void SimulateNet(NET* Net, INT* Input, INT* Target, BOOL Training, BOOL Protocoling)
{
  INT Output[M];
  
  SetInput(Net, Input, Protocoling);
  PropagateNet(Net);
  GetOutput(Net, Output, Protocoling);
  
  ComputeOutputError(Net, Target);
  if (Training)
    AdjustWeights(Net);
}


/******************************************************************************
                                    M A I N
 ******************************************************************************/


void main()
{
  NET  Net;
  REAL Error;
  BOOL Stop;
  INT  n,m;

  InitializeRandoms();
  GenerateNetwork(&Net);
  RandomWeights(&Net);
  InitializeApplication(&Net);
  
  do {
    Error = 0;
    Stop = TRUE;
    for (n=0; n<NUM_DATA; n++) {
      SimulateNet(&Net, Input[n], Output[n], FALSE, FALSE);
      Error = MAX(Error, Net.Error);
      Stop = Stop AND (Net.Error < Net.Epsilon);
    }
    Error = MAX(Error, Net.Epsilon);
    printf("Training %0.0f%% completed ...\n", (Net.Epsilon / Error) * 100);
    if (NOT Stop) {
      for (m=0; m<10*NUM_DATA; m++) {
        n = RandomEqualINT(0, NUM_DATA-1);      
        SimulateNet(&Net, Input[n], Output[n], TRUE, FALSE);
      }
    }
  } while (NOT Stop);
  
  for (n=0; n<NUM_DATA; n++) {
    SimulateNet(&Net, Input[n], Output[n], FALSE, TRUE);
  }
  
  FinalizeApplication(&Net);
}