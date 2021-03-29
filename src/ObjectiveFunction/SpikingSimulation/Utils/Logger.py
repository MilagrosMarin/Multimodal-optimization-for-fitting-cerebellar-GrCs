# This file should be imported after nest. Otherwise nest might not work as expected
import logging
import socket
import os

#from mpi4py import MPI

log_files = []

handler_list = dict()

formatter = None


def InitializeLogger(name):
    
    # mpirank = MPI.COMM_WORLD.Get_rank()

    class ContextFilter(logging.Filter):
        """
        This is a filter which injects contextual information into the log.
        
        Rather than use actual contextual information, we just use random
        data in this demo.
        """
        def filter(self, record):
            # record.mpiid = mpirank
            record.memuse = self.str_mem()
            return True
        
        def _VmB(self):
            """Private.
            """
            import psutil
            process = psutil.Process(os.getpid())
            mem = process.memory_info()[0] / float(2**20)
            return mem
    
        def str_mem(self):
            """Return a string with the total memuse and swap size in MB
            """
            return "MemTotal:%.0fM"%(self._VmB())


    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    global handler_list
    
    if name not in handler_list.keys():
        handler_list[name] = []

        # Create formatter
        global formatter
        if formatter is None:
            #formatter = logging.Formatter('%(asctime)s - P%(process)s - P%(mpiid)s - %(name)s - %(levelname)s: %(message)s')
            formatter = logging.Formatter('%(asctime)s - P%(process)s - %(memuse)s - %(name)s - %(levelname)s: %(message)s')
    
        # Create stdout handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
    
        # Create filter to add the MPI process id
        filter1 = ContextFilter()
    
        logger.addFilter(filter1)
        logger.addHandler(handler)

def Logger2File(logger, filename):
    # Check if a file handler exists with the same file name
    global log_files
    global formatter
    global handler_list
        
    file_handler = [file_hand['handler'] for file_hand in log_files if file_hand['filename']==filename]
    
    # If there is no file_handler with that name
    if not file_handler:
        # Add the hostname and the PID to the filename
        new_filename = filename + '.' + socket.gethostname() + '.' + str(os.getpid())
        handler = logging.FileHandler(new_filename, mode='w')
        log_files.append(dict({'filename': filename,
                               'handler': handler}))
        file_handler = [handler]
        
        if formatter is None:
            formatter = logging.Formatter('%(asctime)s - P%(mpiid)s - %(memuse)s -%(name)s - %(levelname)s: %(message)s')

        handler.setFormatter(formatter)
        
    for handler in file_handler:
        if handler not in handler_list[logger.name]:
            logger.addHandler(handler)
            handler_list[logger.name].append(handler)
        
    
    
    