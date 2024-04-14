#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h> 
#include "mpi.h"
#include <iostream>
#include "math.h"
#include <time.h>

using namespace std;

int main(int *argc, char **argv) {
	//starting timer
	clock_t t;
	t = clock();

	int tn, mn;

	//MPI setup 
	MPI_Init(argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &mn);
	MPI_Comm_size(MPI_COMM_WORLD, &tn);

	//grid setup
	const int N_div = 10e3;
	const double a = 5.0 * 1.0;
	const double dx = 2.0*a / (N_div - 1);
	const double dt = 2.0*10e-4;
	double pr = 10e-5;

	//arrays initialization
	double KPENlocal[4];
	double KPEN[4];
	double V[N_div+1];
	double G[N_div+1];
	double WF[N_div+1];
	double dWF[N_div+1];

	double WF1[N_div + 1];
	double dWF1[N_div + 1];

	double maxg[1];
	double maxglocal[1];
	maxg[0] = 1.0;
	maxglocal[0] = 0.0;
	MPI_Status stat;
	KPENlocal[0] = 0.0;
	KPENlocal[1] = 0.0;
	KPENlocal[2] = 0.0;
	KPENlocal[3] = 1.0;

	for (int i = 0; i <= N_div; i++) {
		//Wave function start values
		WF[i] = 1.0/sqrt(2.0*a);
		//Wave function derivatives and others start values
		dWF[i] = 0.0;
		G[i] = 0.0;
 		//V array callculation
		V[i] = (-a + (2.0 * a)*i / N_div)*(-a + (2.0 * a)*i / N_div)*0.5; //mw=k=1
	}

	int lb = mn * ((N_div ) / tn);
	int rb = (mn + 1)*((N_div) / tn) + ((mn + 1) / tn)*((N_div) % tn);
	int rbudl = (mn + 1)*((N_div) / tn) + ((mn + 1) / tn)*((N_div) % tn+1);


	//main presision controlling loop
	while (maxg[0] >= pr) {
		//borders transfer
		if (mn < tn - 1) { MPI_Send(WF + (rb-1),1,MPI_DOUBLE,mn+1,33,MPI_COMM_WORLD); }
		if (mn > 0) { MPI_Send(WF +lb, 1, MPI_DOUBLE, mn - 1, 32, MPI_COMM_WORLD); }
		if (mn > 0) { MPI_Recv(WF+(lb-1),1, MPI_DOUBLE, mn - 1, 33, MPI_COMM_WORLD,& stat); }
		if (mn < tn - 1) { MPI_Recv(WF+ rb,1, MPI_DOUBLE, mn + 1, 32, MPI_COMM_WORLD, & stat); }
		//K,P, E calculation
		for (int i = lb; i < rb; i++) {
			KPENlocal[0] += V[i]*WF[i] * WF[i]*dx;
			KPENlocal[1] += 0.5* (WF[i + 1] - WF[i])*(WF[i + 1] - WF[i]) / dx;
		}
		KPENlocal[2] = KPENlocal[0] + KPENlocal[1];
		MPI_Allreduce(KPENlocal,KPEN, 4,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD );
		//gradient field calculation
		for (int i = lb; i < rbudl; i++) {
			if ((mn == tn - 1) && (i == rbudl - 1)) {
				G[i] = 2.0*WF[i] * dx*(V[i] - KPEN[2]) + (WF[i] - WF[i - 1]) / dx; 
			}
			else if ((mn == 0) && (i == lb)) { G[i] = 2.0*WF[i] * dx*(V[i] - KPEN[2]) + (WF[i] - WF[i + 1]) / dx; }
			else { G[i] = 2.0 * WF[i] * dx * (V[i] - KPEN[2]) + (2.0 * WF[i] - WF[i + 1] - WF[i - 1]) / dx; }
			if (G[i] >= maxglocal[0]) { maxglocal[0] = G[i]; }
			//dWF and WF iteration
			dWF[i] = 0.99*(dWF[i]-G[i]*dt);
			WF[i] = WF[i] + dWF[i] * dt;
		}
		MPI_Allreduce(maxglocal,maxg, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		//normalisation
		for (int i = lb; i < rb; i++) {
			KPENlocal[3] += WF[i] * WF[i] * dx;
		}
		MPI_Allreduce(KPENlocal+3, KPEN+3, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		for (int i = lb; i < rbudl; i++) {
			WF[i] = WF[i] / sqrt(KPEN[3]);
		}
		//KPElocal, maxglocal wipe
		for (int i = 0; i <= 3; i++) {
			KPENlocal[i] = 0;
		}
		maxglocal[0]=0.0;
	}
	if (mn==0){
		printf("E = %g \n", KPEN[2]);
	}
	//stopping timer;
	t = clock() - t;
	double time_taken = ((double)t) / CLOCKS_PER_SEC;
	printf("Process № %d has taken %f seconds\n", mn, time_taken);
	//getting a WF csv file
	MPI_Reduce(WF+lb,WF1+lb,rb-lb,MPI_DOUBLE,MPI_SUM,0, MPI_COMM_WORLD);
	MPI_Reduce(dWF + lb, dWF1 + lb, rb - lb, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (mn == 0) {
		FILE* f;
		f = fopen("WF.csv", "w");
		for (int i = 0; i <= N_div; i++) {
			fprintf(f, "%g;%g;%g\n", (-a + (2.0 * a)*i / N_div), WF1[i], dWF1);
		}
		fclose(f);
	}
	MPI_Finalize();
}