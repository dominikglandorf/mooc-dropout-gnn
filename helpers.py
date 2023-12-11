class TemporalLoader:
    def __init__(self, data, start_time, batch_size=32):
        self.data = data
        self.time_order = torch.argsort(self.data.time)
        self.start_time = start_time
        self.start_index = torch.nonzero(self.data.time[self.time_order]>=start_time)[0]
        self.index = self.start_index
        self.batch_size = batch_size
        self.length = len(data.edge_y) - self.start_index

    def __iter__(self):
        return self

    def __len__(self):
        return self.length // self.batch_size

    def reset(self):
        self.index = 0

    def __next__(self):
        if self.index + self.batch_size >= self.length:
            self.index = 0
            raise StopIteration

        # these edges are predicted
        mask = torch.zeros(self.data.edge_index.size(1), dtype=torch.bool).to(device)
        mask[self.time_order[self.index+self.start_index:self.index+self.start_index+self.batch_size]] = 1

        # but add neighbors that may be useful for the prediction = edges containing the same IDs in the time before
        before_mask = torch.zeros(self.data.edge_index.size(1), dtype=torch.bool).to(device)
        before_mask[self.time_order[:self.index+self.start_index]] = 1
        first_time = self.data.time[self.time_order[self.index+self.start_index]]

        edge_nodes = self.data.edge_index[:,mask].unique()
        neighbor_mask = create_mask_in_batches(self.data.edge_index, edge_nodes, 256)

        mask |= (neighbor_mask & before_mask)
        
        batch = create_subset(self.data, mask)
        batch.edge_y[batch.time < first_time] = -1
        self.index += self.batch_size
        return batch