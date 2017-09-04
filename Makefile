yapio : yapio.c
	mpicc -o yapio yapio.c -Wall -Wextra -O2

clean :
	rm -f yapio.o yapio
