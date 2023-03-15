train_set.dataset.z_score = (self.mean, self.std)
        # self.test_set.dataset.z_score = (self.mean, self.std)

        # self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        # self.test_loader = DataLoader(self.test_set, batch_size=len(self.test_set), shuffle=False)

        # train_data = next(iter(self.train_loader))