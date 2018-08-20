module init
    use config
    use util
    use feature_config
    use io, only : read_natm, read_config, read_nfeatures, read_features, info_features, error
    use feature_util, only : twobody_features_present,threebody_features_present

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


            !* total number of net weights
            nwght = total_num_weights() 

            !* zero net weights
            net_weights%hl1 = 0.0d0 
            net_weights%hl2 = 0.0d0 
            net_weights%hl3 = 0.0d0 

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

            integer :: seed,conf,atm,ntot,ww
            logical :: need_to_deallocate = .false.
            real(8) :: av_x(1:D)
            
            !* random seed
            call system_clock(seed)
            
            !* initialise
            call srand(seed)
          
            !* total number of atoms for average
            ntot = 0
            av_x = 0.0d0

            do conf=1,data_sets(1)%nconf,1
                ntot = ntot + data_sets(1)%configs(conf)%n
                do atm=1,data_sets(1)%configs(conf)%n
                    av_x = av_x + data_sets(1)%configs(conf)%x(2:,atm)
                end do
            end do
            av_x = av_x / dble(ntot)

            !* feature weights
            call random_number(net_weights%hl1(:,:))
            call random_number(net_weights%hl2(:,:))
            call random_number(net_weights%hl3(:))
            net_weights%hl1(:,:) = ( net_weights%hl1(:,:)-0.5d0)
            net_weights%hl2(:,:) = ( net_weights%hl2(:,:)-0.5d0)
            net_weights%hl3(:)   = ( net_weights%hl3(:)  -0.5d0)

            do ww=1,net_dim%hl1
                net_weights%hl1(ww,1:) = net_weights%hl1(ww,1:) / av_x
            end do

            if (.not.allocated(net_units%a%hl1)) then
                allocate(net_units%a%hl1(net_dim%hl1,data_sets(1)%configs(1)%n))
                need_to_deallocate = .true.
            end if

            ! forward prop
            call dgemm('n','n',net_dim%hl1,data_sets(1)%configs(1)%n,&
                    &D+1,1.0d0,net_weights%hl1,net_dim%hl1,&
                    &data_sets(1)%configs(1)%x,D+1,0.0d0,net_units%a%hl1,net_dim%hl1)

            if (need_to_deallocate) then
                deallocate(net_units%a%hl1)
            end if

            net_weights%hl2(:,:) =  net_weights%hl2(:,:)*1.000d0
            net_weights%hl3(:)   =  net_weights%hl3(:)*1.000d0

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

            atom_neigh_info_needs_updating = .true.

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
            use feature_util, only : performance_option_Nbody_rcut_applies,get_num_fourier_weights
            !use features, only : check_performance_criteria

            implicit none

            character(len=1024),intent(in) :: filepath
            integer :: ft
            integer,allocatable :: num_w(:)

            if (feature_params%num_features.ne.0) then
                !* overwrite previous features
                deallocate(feature_params%info)
            end if

            feature_params%num_features = read_nfeatures(filepath)
            D = feature_params%num_features

            allocate(feature_params%info(feature_params%num_features))

            call read_features(filepath)

            !* check if any performance criteria apply
            if (twobody_features_present()) then                
                if (performance_option_Nbody_rcut_applies(2)) then
                    call activate_performance_option("twobody_rcut")
                end if
            end if
            if (threebody_features_present()) then
                if (performance_option_Nbody_rcut_applies(3)) then
                    call activate_performance_option("threebody_rcut")
                end if
            end if

            !call check_performance_criteria()

            !* pre-compute two/three body type
            do ft=1,feature_params%num_features,1
                if (feature_IsTwoBody(feature_params%info(ft)%ftype)) then
                    feature_params%info(ft)%is_twobody = .true.
                else if (feature_IsThreeBody(feature_params%info(ft)%ftype)) then
                    feature_params%info(ft)%is_threebody = .true.
                end if
            end do

            !* check if 2body fourier feats are present and if all have same length
            if (get_num_fourier_weights(num_w)) then
                !* all fourier feats should have same number of weights
                if(all(num_w.eq.num_w(1))) then
                    call activate_performance_option("equal_fourier_cardinality")  
                end if
            end if
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
            else if (init_type.le.3) then
                set_lim1 = init_type
                set_lim2 = init_type
            else
                call error("init_feature_vectors","unsupported arg value for init_type")
            end if

            do set_type = set_lim1,set_lim2,1
                do conf=1,data_sets(set_type)%nconf,1
                    if (.not.speedup_applies("keep_all_neigh_info")) then
                        if (allocated(data_sets(set_type)%configs(conf)%x)) then
                            deallocate(data_sets(set_type)%configs(conf)%x)
                        end if
                        if (allocated(data_sets(set_type)%configs(conf)%x_deriv)) then
                            deallocate(data_sets(set_type)%configs(conf)%x_deriv)
                        end if
                   
                    end if 
                   
                    if (.not.allocated(data_sets(set_type)%configs(conf)%x)) then 
                        !* feature vector
                        allocate(data_sets(set_type)%configs(conf)%x(D+1,&
                                &data_sets(set_type)%configs(conf)%n))
                    end if
                    if (.not.allocated(data_sets(set_type)%configs(conf)%x_deriv)) then
                        !* feature derivative type
                        allocate(data_sets(set_type)%configs(conf)%x_deriv(D,&
                                &data_sets(set_type)%configs(conf)%n))
                    end if
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

        subroutine allocate_units(set_type,conf)
            implicit none

            integer,intent(in) :: set_type,conf

            !* scratch
            integer :: natm

            natm = data_sets(set_type)%configs(conf)%n

            if (allocated(net_units%a%hl1))  deallocate(net_units%a%hl1)
            if (allocated(net_units%a%hl2))  deallocate(net_units%a%hl2)
            if (allocated(net_units%a_deriv%hl1))  deallocate(net_units%a_deriv%hl1)
            if (allocated(net_units%a_deriv%hl2))  deallocate(net_units%a_deriv%hl2)
            if (allocated(net_units%z%hl1))  deallocate(net_units%z%hl1)
            if (allocated(net_units%z%hl2))  deallocate(net_units%z%hl2)
            if (allocated(net_units%delta%hl1))  deallocate(net_units%delta%hl1)
            if (allocated(net_units%delta%hl2))  deallocate(net_units%delta%hl2)

            !* layer 1
            allocate(net_units%a%hl1(1:net_dim%hl1,1:natm)) 
            allocate(net_units%a_deriv%hl1(1:net_dim%hl1,1:natm))
            allocate(net_units%z%hl1(0:net_dim%hl1,1:natm))
            allocate(net_units%delta%hl1(1:net_dim%hl1,1:natm))
            
            !* layer 2
            allocate(net_units%a%hl2(1:net_dim%hl2,1:natm)) 
            allocate(net_units%a_deriv%hl2(1:net_dim%hl2,1:natm))
            allocate(net_units%z%hl2(0:net_dim%hl2,1:natm))
            allocate(net_units%delta%hl2(1:net_dim%hl2,1:natm))
        end subroutine allocate_units

        subroutine init_set_neigh_info(set_type)
            !* initialise neighbour info for all confs in given set

            implicit none

            integer,intent(in) :: set_type

            if (allocated(set_neigh_info)) then
                deallocate(set_neigh_info)
            end if

            allocate(set_neigh_info(data_sets(set_type)%nconf))
        end subroutine init_set_neigh_info

        subroutine write_net_to_disk(filepath)
            implicit none

            character(len=1024),intent(in) :: filepath

            integer :: funit=5
            real(8),allocatable :: weights_array(:)

            open(unit=funit,file=filepath,action='write')

            !* nodes_layer1 nodes_layer2 activation_function number_features
            write(unit=funit,fmt=*) net_dim%hl1,net_dim%hl2,nlf,feature_params%num_features

            !* concacenate weights into 1d array
            allocate(weights_array(total_num_weights()))

            call parse_structure_to_array(net_weights,weights_array)
            write(unit=funit,fmt=*) weights_array
    
            close(unit=funit)
        end subroutine write_net_to_disk

        subroutine init_net_from_disk(filepath)
            implicit none

            character(len=1024),intent(in) :: filepath

            integer :: funit=5,iostat
            real(8),allocatable :: flat_weights(:)

            open(unit=funit,file=filepath,action='read',iostat=iostat)
            if (iostat.ne.0) then
                write(*,*) 'file :',filepath,'could not be found'
                call exit(0)
            end if

            read(unit=funit,fmt=*) net_dim%hl1,net_dim%hl2,nlf,D
            
            !* initialise weights mem
            call allocate_weights(net_weights)
            
            allocate(flat_weights(total_num_weights()))

            read(unit=funit,fmt=*) flat_weights

            !* parse into nn weights structure
            call parse_array_to_structure(flat_weights,net_weights)
            call allocate_weights_nobiasT(net_weights_nobiasT)
            call copy_weights_to_nobiasT()

            close(unit=funit)
        end subroutine init_net_from_disk

end module init
