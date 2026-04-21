from examples.core_interference.simulator.sim3 import *
import numpy as np


# Simulation setup
#random.seed(0) # For reproducible results

class Experiment:
    def __init__(self,
        num_banks = 4,
        num_addr = 20,
            ):
        self.num_banks = num_banks
        self.num_addr  = num_addr 
        self.num_rows = self.num_addr//16+1
        self.ddr_stats = {}
        self.time_values = {'core0':[0],'core1':[0]}
        # Instantiate the DDR Memory
        self.ddr_memory_physical = DDRMemory(num_banks=self.num_banks)

        # Instantiate the DDR Memory Controller, connected to the physical DDR
        self.ddr_controller = DDRMemoryController(
            self.ddr_memory_physical,
            tRCD=15,    # Row to Column Delay
            tRP=15,     # Row Precharge
            tCAS=15,    # Column Access Strobe latency
            tRC=30,     # Row Cycle time
            tWR=15,     # Write Recovery Time
            tRTP=8,     # Read to Precharge Time
            tCCD=4)     # Column to Column Delay

        # Create interconnect, connected to the DDR Memory Controller
        self.interconnect = Interconnect(self.ddr_controller, delay=5, bandwidth=4)

        # Create cache configurations
        l1_conf = {'size': 32, 'line_size': 4, 'assoc': 2}
        l2_conf = {'size': 512, 'line_size': 4, 'assoc': 16}


        # Create shared L2 Cache, connected to the Interconnect
        shared_l2 = CacheLevel("L2", core_id="anycore", memory=self.interconnect, **l2_conf)

        self.num_set = shared_l2.num_sets 
        self._index = shared_l2._index
        #shared_l2.num_tags = shared_l2._tag(self.num_addr)
        #shared_l2.tab_miss = self.

        # Create Core-specific Multi-Level Caches, connected to the shared L2
        self.mem_core0 = MultiLevelCache(0, l1_conf, shared_l2)
        self.mem_core1 = MultiLevelCache(1, l1_conf, shared_l2)

        # Create cores
        self.core0 = Core(0, self.mem_core0)
        self.core1 = Core(1, self.mem_core1)
    def add_time_values(self,values:dict[list]):
        if type(values['core0'])!=type(None):
                self.time_values['core0'].append(values['core0'])
        if type(values['core1'])!=type(None):
                self.time_values['core1'].append(values['core1'])
    def add_values(self,ddr_stats):
        if type(ddr_stats)!=type(None):
            for key in ddr_stats:
                if key in self.ddr_stats and key!='completion_time':
                    self.ddr_stats[key].append(ddr_stats[key])
                elif key =='completion_time':
                    self.time_values[['core0','core1'][ddr_stats['core']]].append(ddr_stats[key])
                else:
                    self.ddr_stats[key]=[ddr_stats[key]]
    def load_instr(self, core0_inst, core1_inst):
        self.core0.load_instr(core0_inst)
        self.core1.load_instr(core1_inst)

    def simulate(self, cycles,display_stats=False):
        GlobalVar.global_cycle = 0
        for cycle in range(cycles):
            # /!\ All components tick at the same frequency
            time0 = self.core0.tick()
            time1 = self.core1.tick()
            self.interconnect.tick()
            ddr_stats = self.ddr_controller.tick()
            self.add_values(ddr_stats)
            self.add_time_values({'core0':time0,'core1':time1})
            self.ddr_memory_physical.tick()
            # Update global clock (shared variable)
            GlobalVar.global_cycle+=1

        self.cache_stats_core_0 = self.mem_core0.stats()
        self.cache_stats_core_1 = self.mem_core1.stats()
        self.reorder()
        if display_stats:
            # Report results
            print("\n--- Simulation Stats ---")
            dict0 = self.cache_stats_core_0['L1'].copy()
            dict1 = self.cache_stats_core_1['L1'].copy()
            display_dict = self.cache_stats_core_0['L2'].copy()
            del dict0['cache_miss_detailled']
            del dict1['cache_miss_detailled']
            del display_dict['cache_miss_detailled']
            print('core0',dict0)
            print('core1',dict1)
            print('shared cache L2',display_dict)
            print('ddr hits', self.hits_tab)
            print('ddr miss', self.miss_tab)
        return self.output_data()
    def reorder(self):
        hits = np.zeros(self.num_banks)
        miss = np.zeros(self.num_banks)
        type2id = lambda type_:0 if type_=='read' else 1
        self.hits_tab = np.zeros((2,self.num_rows,self.num_banks))
        self.miss_tab = np.zeros((2,self.num_rows,self.num_banks))
        if 'row' in self.ddr_stats:
            for j in range(len(self.ddr_stats['row'])):
                if self.ddr_stats['status'][j]=='ROW MISS':
                    miss[self.ddr_stats['bank'][j]] +=1
                    self.miss_tab[type2id(self.ddr_stats['current_type'][j]),self.ddr_stats['row'][j],self.ddr_stats['bank'][j]] +=1
                else:
                    hits[self.ddr_stats['bank']] +=1
                    self.hits_tab[type2id(self.ddr_stats['current_type'][j]),self.ddr_stats['row'][j],self.ddr_stats['bank'][j]] +=1


        denominator = miss + hits
        denominator[denominator==0] = -1
        self.ratios = miss/(denominator)
        self.ratios[self.ratios<=0] = -1
        #self.analyze_interference_events = analyze_shared_resource_contention()
        if (np.sum(miss)+np.sum(hits))==0:
            self.miss_ratio_global =-1
        else:
            self.miss_ratio_global = np.sum(miss)/(np.sum(miss)+np.sum(hits))

        denominator_tab  = self.miss_tab + self.hits_tab

        denominator_tab_read = denominator_tab[0]
        denominator_tab_write = denominator_tab[1]

        denominator_tab = denominator_tab.sum(axis=0)

        denominator_tab[denominator_tab==0] = -1
        denominator_tab_read[denominator_tab_read==0] = -1
        denominator_tab_write[denominator_tab_write==0] = -1

        #self.ratios_tab = self.miss_tab/(denominator_tab)
        self.ratios_tab = self.miss_tab.sum(axis=0)/(denominator_tab)
        self.ratios_tab[self.ratios_tab<0] = -1

        self.ratios_tab_read = self.miss_tab[0]/(denominator_tab_read)
        self.ratios_tab_write = self.miss_tab[1]/(denominator_tab_write)
        self.ratios_tab_read[self.ratios_tab_read<0] = -1
        self.ratios_tab_write[self.ratios_tab_write<0] = -1

        #details for shared cache miss ratio
        #self.cache_miss_ratio_tab = sel
    def output_data(self):
        return {'time_core0':max(self.time_values['core0']),
                'time_core1':max(self.time_values['core1']),
                'miss_ratios_detailled':self.ratios_tab,
                'miss_ratios_detailled_read':self.ratios_tab_read,
                'miss_ratios_detailled_write':self.ratios_tab_write,
                'miss_ratios_global': self.ratios,
                'miss':self.miss_tab.sum(axis=0),
                'hits':self.hits_tab.sum(axis=0),
                'miss_read':self.miss_tab[0],
                'hits_read':self.hits_tab[0],
                'miss_write':self.miss_tab[1],
                'hits_write':self.hits_tab[1],
                'L2_miss':self.cache_stats_core_0['L2']['misses'],
                'L2_hit': self.cache_stats_core_0['L2']['hits'],
                'L2_miss_write':self.cache_stats_core_0['L2']['misses_write'],
                'L2_miss_read':self.cache_stats_core_0['L2']['misses_read'],
                'L2_hit_write':self.cache_stats_core_0['L2']['hits_write'],
                'L2_hit_read':self.cache_stats_core_0['L2']['hits_read'],
                #'shared_ressource_events':GlobalVar.shared_resource_events,
                }
class Env:
    def __init__(self,cycles,
                 num_banks = 4,
                 num_addr = 20,
                ):
        self.num_banks = num_banks
        self.num_addr  = num_addr 
        self.num_rows = self.num_addr//16#+1
        self.cycles = cycles
    def __call__(self, parameter:dict)->dict:
        program  = Experiment(num_banks=self.num_banks,num_addr=self.num_addr)
        program0 = Experiment(num_banks=self.num_banks,num_addr=self.num_addr)
        program1 = Experiment(num_banks=self.num_banks,num_addr=self.num_addr)
        program.load_instr(parameter["core0"], parameter["core1"])
        program0.load_instr(parameter["core0"],[])
        program1.load_instr([],parameter["core1"])
        out = {}
        GlobalVar.clear_history()
        out['core0'] = program0.simulate(self.cycles)
        GlobalVar.clear_history()
        out['core1'] = program1.simulate(self.cycles)
        GlobalVar.clear_history()
        out['mutual'] = program.simulate(self.cycles)
        GlobalVar.clear_history()
        del out['core0']['time_core1']
        del out['core1']['time_core0']
        #ddr targets
        out['mutual']['diff_time_core0'] = out['mutual']['time_core0'] - out['core0']['time_core0']
        out['mutual']['diff_time_core1'] = out['mutual']['time_core1'] - out['core1']['time_core1']
        out['mutual']['diff_time'] = out['mutual']['time_core1'] - out['mutual']['time_core0']
        out['mutual']['miss_core0'] = out['mutual']['miss'] - out['core0']['miss']
        out['mutual']['miss_core1'] = out['mutual']['miss'] - out['core1']['miss']
        out['mutual']['hits_core0'] = out['mutual']['hits'] - out['core0']['hits']
        out['mutual']['hits_core1'] = out['mutual']['hits'] - out['core1']['hits']
        out['mutual']['miss_read_core0'] = out['mutual']['miss_read'] - out['core0']['miss_read']
        out['mutual']['miss_read_core1'] = out['mutual']['miss_read'] - out['core1']['miss_read']
        out['mutual']['miss_write_core0'] = out['mutual']['miss_write'] - out['core0']['miss_write']
        out['mutual']['miss_write_core1'] = out['mutual']['miss_write'] - out['core1']['miss_write']
        out['mutual']['hits_read_core0'] = out['mutual']['hits_read'] - out['core0']['hits_read']
        out['mutual']['hits_read_core1'] = out['mutual']['hits_read'] - out['core1']['hits_read']
        out['mutual']['hits_write_core0'] = out['mutual']['hits_write'] - out['core0']['hits_write']
        out['mutual']['hits_write_core1'] = out['mutual']['hits_write'] - out['core1']['hits_write']
        #L2 targets
        out['mutual']['L2_miss_core0'] = out['mutual']['L2_miss'] - out['core0']['L2_miss']
        out['mutual']['L2_hit_core0'] = out['mutual']['L2_hit'] - out['core0']['L2_hit']
        out['mutual']['L2_miss_core1'] = out['mutual']['L2_miss'] - out['core1']['L2_miss']
        out['mutual']['L2_hit_core1'] = out['mutual']['L2_hit'] - out['core1']['L2_hit']


        out['mutual']['L2_miss_read_core0'] = out['mutual']['L2_miss_read'] - out['core0']['L2_miss_read']
        out['mutual']['L2_hit_read_core0'] = out['mutual']['L2_hit_read'] - out['core0']['L2_hit_read']
        out['mutual']['L2_miss_read_core1'] = out['mutual']['L2_miss_read'] - out['core1']['L2_miss_read']
        out['mutual']['L2_hit_read_core1'] = out['mutual']['L2_hit_read'] - out['core1']['L2_hit_read']
        out['mutual']['L2_miss_write_core0'] = out['mutual']['L2_miss_write'] - out['core0']['L2_miss_write']
        out['mutual']['L2_hit_write_core0'] = out['mutual']['L2_hit_write'] - out['core0']['L2_hit_write']
        out['mutual']['L2_miss_write_core1'] = out['mutual']['L2_miss_write'] - out['core1']['L2_miss_write']
        out['mutual']['L2_hit_write_core1'] = out['mutual']['L2_hit_write'] - out['core1']['L2_hit_write']
        return out
