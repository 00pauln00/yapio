yapio : yapio.c
	mpicc -o yapio yapio.c -Wall -Wextra -O2

asan :
	mpicc -g -o yapio yapio.c -Wall -Wextra -fsanitize=address

clean :
	rm -f yapio.o yapio
