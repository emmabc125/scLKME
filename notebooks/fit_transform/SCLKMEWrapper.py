from scanpy.tools._utils import _choose_representation
from sclkme.tl import sketch, kernel_mean_embedding

class SCLKMEWrapper:
    
    def __init__(self, partition_key='label', method="exact", kernel="rbf",n_sketch=512,random_state=42,use_rep='X',n_jobs=-2,sketch_method='random'):
        #parameters
        self.method = method
        self.kernel = kernel
        self.n_sketch = n_sketch
        self.partition_key = partition_key
        self.random_state=random_state
        self.use_rep=use_rep
        self.n_jobs=n_jobs
        self.sketch_method=sketch_method
        
        
        #data created upon .fit(X)
        self.X_anchor = None
        self.Den = 0
        self.sketched_ids = None
        
        
    def fit(self,adata,y=None):
        
        adata_c = adata.copy()
        sketch(adata_c, n_sketch=self.n_sketch,use_rep=self.use_rep, method=self.sketch_method)
        self.X_anchor = _choose_representation(adata_c[adata_c.obs['sketch']], use_rep=self.use_rep)
        kernel_mean_embedding(adata_c, 
                                        partition_key=self.partition_key,
                                        X_anchor=self.X_anchor,
                                        use_rep=self.use_rep,
                                        method=self.method,
                                        kernel=self.kernel, 
                                        n_jobs=self.n_jobs)
        self.Den = adata_c.uns['kme']['Den']
        self.sketched_ids=adata_c.obs['sketch']
        
        
    def fit_transform(self,adata,y=None):
        adata_c = adata.copy()
        sketch(adata_c, n_sketch=self.n_sketch,use_rep=self.use_rep, method=self.sketch_method)
        
        self.X_anchor = _choose_representation(adata_c[adata_c.obs['sketch']], use_rep=self.use_rep)
        kernel_mean_embedding(adata_c, 
                                        partition_key=self.partition_key,
                                        X_anchor=self.X_anchor,
                                        use_rep=self.use_rep,
                                        method=self.method,
                                        kernel=self.kernel,
                                        n_jobs=self.n_jobs)
        self.Den = adata_c.uns['kme']['Den']
        self.sketched_ids=adata_c.obs['sketch']

        return adata_c.uns['kme'][self.partition_key+'_kme']
    
        
    def transform(self,adata):
        adata_c = adata.copy()
        kernel_mean_embedding(adata_c, 
                                        partition_key=self.partition_key,
                                        X_anchor=self.X_anchor,
                                        use_rep=self.use_rep,
                                        method=self.method,
                                        kernel=self.kernel,
                                        kernel_kwds={'Den':self.Den},
                                        n_jobs=self.n_jobs)
        return adata_c.uns['kme'][self.partition_key+'_kme']
    