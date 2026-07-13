include ../../../System/makefile.h

SOURCES = broyden.c electros.c
###################################
SOURCES += electros_monopole.c
###################################
SOURCES += freestrucs.c get_integral_idxs.c input.c parseadt.c parsebdt.c magnetic.c manip_bdttabs.c states.c tables.c

OBJECTS = ${SOURCES:.c=.o}

../../../lib/libstates.a: $(OBJECTS)
	ar -r ../../../lib/libstates.a $(OBJECTS)
	ranlib ../../../lib/libstates.a

clean:
	rm -f *.o *~ 

tar:
	tar -cvf libstates.tar *.c *.h makefile
	gzip libstates.tar
	mv libstates.tar.gz libstates.tgz

%.o: %.c states.h
	$(CC) -c $(CFLAGS) $(DEFS) -o $@ $<
