CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -pedantic -pthread -fopenmp -g -fsanitize=address -fsanitize=undefined

COMMON_OBJS = LinReg.o datasetloader.o matrix.o LogReg.o evaluation.o nn.o

all: LogReg-demo nn-demo LinReg-demo

LogReg-demo: $(COMMON_OBJS) LogReg-demo.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

LinReg-demo: $(COMMON_OBJS) LinReg-demo.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

nn-demo: $(COMMON_OBJS) nn-demo.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

matrix.o: matrix.c
	$(CC) $(CFLAGS) -c matrix.c

LinReg.o: LinReg.c
	$(CC) $(CFLAGS) -c LinReg.c

LogReg.o: LogReg.c
	$(CC) $(CFLAGS) -c LogReg.c

datasetloader.o: datasetloader.c
	$(CC) $(CFLAGS) -c datasetloader.c

LogReg-demo.o: LogReg-demo.c
	$(CC) $(CFLAGS) -c LogReg-demo.c

LinReg-demo.o: LinReg-demo.c
	$(CC) $(CFLAGS) -c LinReg-demo.c

nn-demo.o: nn-demo.c
	$(CC) $(CFLAGS) -c nn-demo.c

evaluation.o: evaluation.c
	$(CC) $(CFLAGS) -c evaluation.c

nn.o: nn.c
	$(CC) $(CFLAGS) -c nn.c

clean:
	rm -f *.o LogReg-demo nn-demo LinReg-demo