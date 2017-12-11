module init
    use config
    use util
    use feature_config
    use io, only : read_natm, read_config, read_nfeatures, read_features, info_features, error

    implicit none
    
    contains
        subroutine initialise_net(num_nodes,nlf_type,feat_D)
            implicit none

            integer,intent(in) :: num_nodes(1:2),nlf_type,feat_D

            !* feature space dimension
            D = feat_D

            !* nodes in each hidden layer
            net_dim%hl1 = num_nodes(1) 
            net_dim%hl2 = num_nodes(2) 

            !* type of nonlinear activation func
            nlf = nlf_type

            call allocate_weights(net_weights)
            call allocate_weights_nobiasT(net_weights_nobiasT)
            call allocate_weights(dydw)
           
            if (allocated(net_units%a%hl1)) then
                deallocate(net_units%a%hl1)
                deallocate(net_units%a%hl2)
                deallocate(net_units%a_deriv%hl1)
                deallocate(net_units%a_deriv%hl2)
            end if

            allocate(net_units%a%hl1(net_dim%hl1))
            allocate(net_units%a%hl2(net_dim%hl2))
            allocate(net_units%a_deriv%hl1(net_dim%hl1))
            allocate(net_units%a_deriv%hl2(net_dim%hl2))
           
            if (allocated(net_units%z%hl1)) then
                deallocate(net_units%z%hl1)
                deallocate(net_units%z%hl2)
            end if
            
            !* include null value for bias
            allocate(net_units%z%hl1(0:net_dim%hl1))
            allocate(net_units%z%hl2(0:net_dim%hl2))
           
            if (allocated(net_units%delta%hl1)) then
                deallocate(net_units%delta%hl1)
                deallocate(net_units%delta%hl2)
            end if
            
            allocate(net_units%delta%hl1(net_dim%hl1))
            allocate(net_units%delta%hl2(net_dim%hl2))

            !* total number of net weights
            nwght = total_num_weights() 

            !* initialise NN weights
            call random_weights()

            call check_input()
        end subroutine initialise_net

        subroutine init_loss(k_energy,k_forces,k_reglrn,norm_type)
            implicit none

            real(8),intent(in) :: k_energy,k_forces,k_reglrn
            integer,intent(in) :: norm_type

            loss_const_energy = k_energy
            loss_const_forces = k_forces
            loss_const_reglrn = k_reglrn

            loss_norm_type = norm_type
        end subroutine
  
        subroutine allocate_weights(weights_in)
            implicit none

            type(weights),intent(inout) :: weights_in

            if(allocated(weights_in%hl1)) then
                call deallocate_weights(weights_in)
            end if

            !* include bias for each node
            
            allocate(weights_in%hl1(1:net_dim%hl1,0:D))
            allocate(weights_in%hl2(1:net_dim%hl2,0:net_dim%hl1))
            allocate(weights_in%hl3(0:net_dim%hl2))
            
            !allocate(weights_in%hl1(D+1,net_dim%hl1))
            !allocate(weights_in%hl2(net_dim%hl1+1,net_dim%hl2))
            !allocate(weights_in%hl3(net_dim%hl2+1))
        end subroutine
        
        subroutine allocate_weights_nobiasT(weights_in)
            implicit none

            type(weights),intent(inout) :: weights_in

            if(allocated(weights_in%hl1)) then
                call deallocate_weights(weights_in)
            end if

            !* do NOT include bias
            allocate(weights_in%hl1(1:D,1:net_dim%hl1))
            allocate(weights_in%hl2(1:net_dim%hl1,1:net_dim%hl2))
            allocate(weights_in%hl3(1:net_dim%hl2))
            
            !allocate(weights_in%hl1(D+1,net_dim%hl1))
            !allocate(weights_in%hl2(net_dim%hl1+1,net_dim%hl2))
            !allocate(weights_in%hl3(net_dim%hl2+1))
        end subroutine allocate_weights_nobiasT

        subroutine allocate_d2ydxdw_mem(conf,set_type,d2ydxdw)
            implicit none

            integer,intent(in) :: conf,set_type
            type(weights),intent(inout),allocatable :: d2ydxdw(:,:)

            integer :: natm,ii,kk

            natm = data_sets(set_type)%configs(conf)%n

            if (allocated(d2ydxdw)) then
                deallocate(d2ydxdw)
            end if

            allocate(d2ydxdw(natm,D))

            do kk=1,D
                do ii=1,natm
                    call allocate_weights(d2ydxdw(ii,kk))
                end do
            end do
        end subroutine allocate_d2ydxdw_mem

        subroutine deallocate_weights(weights_in)
            implicit none
            
            type(weights),intent(inout) :: weights_in

            deallocate(weights_in%hl1)
            deallocate(weights_in%hl2)
            deallocate(weights_in%hl3)
        end subroutine deallocate_weights

        subroutine initialise_set(set_type,nconf,ntot,slice_idxs,xin,fin,ein)
            use io, only : error
            
            implicit none

            integer,intent(in) :: set_type  !* 1 = train , 2 = test
            integer,intent(in) :: nconf,ntot,slice_idxs(1:2,1:nconf)
            real(8),intent(in) :: xin(1:D,1:ntot),fin(1:3,1:ntot)
            real(8),intent(in) :: ein(1:nconf)
          
            !* scratch
            integer :: ii,natm,idx1,idx2
            
            if ( (set_type.lt.1).or.(set_type.gt.2) ) then
                call error("initialise_set","unsupported set_type")
            end if

            !* allocate memory for structures
            allocate(data_sets(set_type)%configs(nconf))

            !* number of structures in set
            data_sets(set_type)%nconf = nconf

            !* parse features
            do ii=1,nconf,1
                !* number atoms
                natm = slice_idxs(2,ii) - slice_idxs(1,ii) + 1

                !* initial and final atoms
                idx1 = slice_idxs(1,ii)
                idx2 = slice_idxs(2,ii)

                data_sets(set_type)%configs(ii)%n = natm
                
                !* include null dimension for biases
                allocate(data_sets(set_type)%configs(ii)%x(D+1,natm))
                allocate(data_sets(set_type)%configs(ii)%ref_fi(3,natm))
                allocate(data_sets(set_type)%configs(ii)%current_ei(natm))
                allocate(data_sets(set_type)%configs(ii)%current_fi(3,natm))

                !* for biases
                data_sets(set_type)%configs(ii)%x(1,:) = 1.0d0
                
                !* features
                data_sets(set_type)%configs(ii)%x(2:,1:natm) = xin(:,idx1:idx2)

                !* ref: energy
                data_sets(set_type)%configs(ii)%ref_energy = ein(ii)

                !* ref: forces
                data_sets(set_type)%configs(ii)%ref_fi(1:3,:) = fin(:,idx1:idx2)
            end do
        end subroutine

        subroutine random_weights()
            ! ==================================================!
            ! first element of each linear model is the bias    !
            ! ==================================================!
            
            implicit none

            integer :: seed

            !* random seed
            call system_clock(seed)
            
            !* initialise
            call srand(seed)
          
            !* feature weights
            call random_number(net_weights%hl1(:,:))
            call random_number(net_weights%hl2(:,:))
            call random_number(net_weights%hl3(:))

            net_weights%hl1(:,:) = ( net_weights%hl1(:,:)-0.5d0)*0.001d0
            net_weights%hl2(:,:) = ( net_weights%hl2(:,:)-0.5d0)*0.001d0
            net_weights%hl3(:)   = ( net_weights%hl3(:)  -0.5d0)*0.001d0

            !* set biases to zero
            net_weights%hl1(:,0) = 0.0d0
            net_weights%hl2(:,0) = 0.0d0
            net_weights%hl3(0)  = 0.0d0

            !* transpose with no bias
            call copy_weights_to_nobiasT()
        end subroutine random_weights

        subroutine check_input()
            use io, only : error
            
            implicit none

            !* check nlf type
            if ( (nlf.le.0).or.(nlf.ge.3) ) then
                call error("check_input","nonlinear function type unsupported")
            end if
        end subroutine

        subroutine init_configs_from_disk(files,set_type)
            implicit none

            !* args
            character(len=1024),dimension(:),intent(in) :: files
            integer,intent(in) :: set_type

            !* scratch
            integer :: conf,natm,dim(1:1)

            if ( data_sets(set_type)%nconf.ne.0 ) then
                !* need to deallocate now deprecated data
                deallocate(data_sets(set_type)%configs)
            end if
           
            dim = shape(files)
            
            !* number of configurations in this data set
            data_sets(set_type)%nconf = dim(1)/1024
            
            !* start from scratch
            allocate(data_sets(set_type)%configs(data_sets(set_type)%nconf))
            
            do conf=1,data_sets(set_type)%nconf,1
                !* number of atoms
                data_sets(set_type)%configs(conf)%n = read_natm(files(conf))
                
                natm = data_sets(set_type)%configs(conf)%n 
                
                allocate(data_sets(set_type)%configs(conf)%r(3,natm))
                allocate(data_sets(set_type)%configs(conf)%z(natm))
                allocate(data_sets(set_type)%configs(conf)%current_ei(natm))
                allocate(data_sets(set_type)%configs(conf)%current_fi(3,natm))
                allocate(data_sets(set_type)%configs(conf)%ref_fi(3,natm))
                !* parse data into fortran data struct.
                call read_config(set_type,conf,files(conf))
            end do
        end subroutine init_configs_from_disk

        subroutine init_features_from_disk(filepath)
            implicit none

            character(len=1024),intent(in) :: filepath

            if (feature_params%num_features.ne.0) then
                !* overwrite previous features
                deallocate(feature_params%info)
            end if

            feature_params%num_features = read_nfeatures(filepath)
            D = feature_params%num_features

            allocate(feature_params%info(feature_params%num_features))

            call read_features(filepath)
        end subroutine init_features_from_disk

        subroutine init_feature_vectors(init_type)
            !=======================================================!
            ! Configuration data and feature information must       !
            ! already be set                                        !
            !                                                       !
            ! Parameters                                            !
            ! ----------                                            !
            ! init_type : int, allowed values = 0,1,2               !
            !     If init_type=0, initialise feature vectors for    !
            !     both data sets, if init_type=1, initialise only   !
            !     for training data, else for test data             !
            !=======================================================!
            
            implicit none

            !* args
            integer,intent(in) :: init_type
            
            !* scratch 
            integer :: set_type,set_lim1,set_lim2,conf
   
            set_lim1 = -1
            set_lim2 = -1
            if (feature_params%num_features.le.0) then
                call error("init_feature_vectors","features have not been set")
            else
                do set_type=1,2
                    if ( (init_type.eq.0).or.(init_type.eq.set_type) ) then
                        if (data_sets(set_type)%nconf.eq.0) then
                            call error("init_feature_vectors","data for set has not been initialised")
                        end if
                    end if
                end do !* loop over sets

                if (D.ne.feature_params%num_features) then
                    call error("init_feature_vectors","mismatch has occured in number of features")
                end if
            end if

            if (init_type.eq.0) then
                set_lim1 = 1
                set_lim2 = 2
            else if (init_type.le.2) then
                set_lim1 = init_type
                set_lim2 = init_type
            else
                call error("init_feature_vectors","unsupported arg value for init_type")
            end if

            do set_type = set_lim1,set_lim2,1
                do conf=1,data_sets(set_type)%nconf,1
                    !* check if arrays need deallocating
                    if (allocated(data_sets(set_type)%configs(conf)%x)) then
                        deallocate(data_sets(set_type)%configs(conf)%x)
                    end if
                    if (allocated(data_sets(set_type)%configs(conf)%x_deriv)) then
                        deallocate(data_sets(set_type)%configs(conf)%x_deriv)
                    end if
                    
                    !* feature vector
                    allocate(data_sets(set_type)%configs(conf)%x(D+1,data_sets(set_type)%configs(conf)%n))
                    
                    !* feature derivative type
                    allocate(data_sets(set_type)%configs(conf)%x_deriv(D,data_sets(set_type)%configs(conf)%n))
                end do !* end loop over confs
            end do !* end loop over data sets

        end subroutine init_feature_vectors

        subroutine finalize()
            implicit none

            deallocate(net_weights%hl1)
            deallocate(net_weights%hl2)
            deallocate(net_weights%hl3)
            deallocate(net_units%a%hl1)
            deallocate(net_units%a%hl2)
            deallocate(net_units%z%hl1)
            deallocate(net_units%z%hl2)
            deallocate(net_units%delta%hl1)
            deallocate(net_units%delta%hl2)
            deallocate(data_sets(1)%configs)
            deallocate(data_sets(2)%configs)
        end subroutine finalize

end module init
